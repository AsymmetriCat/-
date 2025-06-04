# 导入所需模块
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CelebA
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss
from PIL import Image

# 预处理变换
image_transform = transforms.Compose([
    transforms.Resize(128),          # 调整较短边到128
    transforms.CenterCrop(128),      # 裁剪图像尺寸为128*128
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(            # 采用ImageNet归一化
        mean=[0.485, 0.456, 0.406],  # 计算公式: normalized_channels=(input_channels-mean)/std
        std= [0.229, 0.224, 0.225]
    )
])

# 定义训练集与测试集
training_set = CelebA(
    root='./data',
    split='train',
    target_type='attr',
    download=False,
    transform=image_transform
)
test_set = CelebA(
    root='./data',
    split='test',
    target_type='attr',
    download=False,
    transform=image_transform
)

# 定义数据加载器
batch_size = 64
training_dataloader = DataLoader(training_set, batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size, shuffle=False)

# 设置运算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"实验所使用的运算设备: {device}")

# 多层感知机
class SimpleMLP(nn.Module):
    def __init__(self, num_classes=40, input_dim=3*128*128, hidden_dim=[1024, 512, 256]):
        super().__init__()
        self.flatten = nn.Flatten()     # 对输入数据进行展平化处理
        self.features = nn.Sequential(  # 特征提取
            # 第一层
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),  # 在激活函数前进行批量归一化
            nn.ReLU(),
            nn.Dropout(0.3),                # 训练时随机丢弃部分神经元以减少过拟合
            # 第二层
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 第三层
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(  # 分类
            nn.Linear(hidden_dim[2], num_classes)
        )
    # 定义前向传播过程
    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

# 简易卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.features = nn.Sequential(  # 特征提取
            # 第一块卷积    输入: 3*128*128
            nn.Conv2d(            # 定义卷积层
                in_channels=3,    # 输入RGB图像
                out_channels=32,  # 设置卷积核数为32
                kernel_size=3,    # 卷积核3*3大小
                padding=1         # 边界各填充1像素
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(         # 最大池化
                kernel_size=2,    # 池化窗口尺寸
                stride=2          # 步长大小
            ),  # 输出尺寸计算公式: out_size = (in_size-kernel_size+2*padding)/stride + 1
            # 第二块卷积    输入: 32*64*64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # 第三块卷积    输入: 64*32*32
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # 第四块卷积    输入: 128*16*16 最终输出: 256*8*8
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256*8*8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    # 定义前向传播
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# 定义损失函数
loss_func = nn.BCEWithLogitsLoss()
# 模型
mlp_model = SimpleMLP().to(device)
cnn_model = SimpleCNN().to(device)
# 优化器 初始学习率设为0.001, L2正则化强度设为0.0001
mlp_optim = torch.optim.Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
cnn_optim = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
# 学习率调度器 采用余弦退火使得学习率周期性变化, 将一周期设为30次迭代
mlp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(mlp_optim, T_max=30)
cnn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optim, T_max=30)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()  # 训练模式
    total_batches = len(dataloader)  # 总批次数
    total_losses = 0  # 总损失值
    num_batches = 0  #已处理的批次数
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.float().to(device)  # 此处y必须要为浮点值, BCEWithLogitsLoss涉及浮点数运算
        optimizer.zero_grad()  # 前向传播前清零梯度
        pred = model(x)  # 预测值
        loss = loss_fn(pred, y)  # 损失值
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度范数裁剪以防梯度爆炸
        optimizer.step()  # 更新网络中的参数
        total_losses += loss.item() # 提取损失的标量值
        num_batches += 1  # 至此处理完一批次数据
        # 每处理完100个批次打印一次信息
        if (batch + 1) % 100 == 0:
            print(f'[{batch+1}/{total_batches}]批次损失值: {loss.item():.6f}, '
                                             f'平均损失值: {total_losses/num_batches:.6f}')
    # 打印本次训练信息
    avg_loss = total_losses / num_batches
    print(f'本次训练总体平均损失值: {avg_loss:.6f}')
    # 返回平均损失
    return avg_loss

# 定义测试函数
def test(dataloader, model, loss_fn, num_classes=40, threshold=0.5):
    model.eval()  # 评估模式
    total_batches = len(dataloader)
    total_losses = 0
    predictions = []  # 预测值列表
    actuals = []  # 真实值列表
    # 测试过程并不涉及反向传播与梯度运算
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.float().to(device)
            pred = model(x)
            total_losses += loss_fn(pred, y).item()
            probability = torch.sigmoid(pred)  # 利用sigmoid激活函数将预测值映射至0到1区间表示概率
            binary = (probability > threshold).float()  # 二值化
            # 张量转换为numpy数组添加至列表, numpy只能在cpu上运算
            predictions.append(binary.cpu().numpy())
            actuals.append(y.cpu().numpy())
    # 评估模型性能
    # 垂直堆叠
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    avg_loss = total_losses / total_batches  # 平均测试损失值
    whole_accuracy = accuracy_score(actuals, predictions)  # 整体准确率
    hamming = hamming_loss(actuals, predictions)  # 汉明损失
    attr_accuracies = []  # 各属性准确率
    for i in range(num_classes):
        attr_accuracy = accuracy_score(actuals[:, i], predictions[:, i])  # 按列索引, 计算该属性准确率
        attr_accuracies.append(attr_accuracy)
    avg_accuracy = np.mean(attr_accuracies)  # 求均值得到平均单属性准确率
    # 打印信息
    print(f'平均测试损失值: {avg_loss:.6f}')
    print(f'整体准确率: {whole_accuracy*100:.2f}%')
    print(f'汉明损失值: {hamming:.6f}')
    print(f'平均单属性准确率: {avg_accuracy*100:.2f}%')
    # 返回信息
    return avg_loss, whole_accuracy, hamming, attr_accuracy

# 定义迭代训练过程
def iteratively_train(training_loader, test_loader, model, loss_fn, optimizer,
                   scheduler, model_path, epochs=30, num_classes=40, threshold=0.5):
    print('开始训练模型:')
    best_loss = float('inf')  # 最佳平均损失值, 初始化为无穷大
    patience = 5  # 耐心值, 如果连续5轮模型没有任何优化, 暂停训练
    patience_counter = 0  # 耐心计数器
    # 训练与测试
    for t in range(epochs):
        print(f'第{t+1}/{epochs}轮:')
        train(training_loader, model, loss_fn, optimizer)  # 本轮训练平均损失值
        test_loss, *_ = test(test_loader, model, loss_fn, num_classes, threshold)  # 测试
        scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']  # 获取默认参数组中的学习率
        print(f'当前学习率: {current_lr:.6f}')
        # 早停机制
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f'保存本轮训练的模型至{model_path}')
        else:
            patience_counter += 1
            print('本轮训练模型没有优化')
            if patience_counter >= patience:
                print(f'连续{patience}轮模型没有优化, 退出迭代训练')
                break
    print('训练结束')

# 读取模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 量化权重

# CelebA的分类标签
celeba_classes = [
    '5点钟胡茬', '弓形眉', '有吸引力', '眼袋', '秃头',
    '刘海', '厚嘴唇', '大鼻子', '黑发', '金发',
    '模糊', '棕发', '浓眉', '胖脸', '双下巴',
    '戴眼镜', '山羊胡', '灰发', '化浓妆', '高颧骨',
    '男性', '嘴微张', '胡子', '细眼', '无胡须',
    '椭圆脸', '苍白皮肤', '尖鼻子', '发际线后移', '红脸颊',
    '鬓角', '微笑', '直发', '波浪发', '戴耳环',
    '戴帽子', '涂口红', '戴项链', '戴领带', '年轻'
]

# 使用模型预测图像属性
def predict_image(model, image_path, threshold=0.5):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加batch维度， 切记要移动到运算设备上
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]  # 第一个样本即为需要预测的图像
            preds = probs > threshold  # 二值
            return probs, preds
    except FileNotFoundError:
        raise FileNotFoundError(f'未找到图片{image_path}')

# 使用模型
def use_model(model, image_path, threshold=0.5):
    try:
        probs, preds = predict_image(model, image_path, threshold)
        print("预测结果:")
        for _, (attr_name, prob, pred) in enumerate(zip(celeba_classes, probs, preds)):
            status = "✓" if pred else "✗"  # 根据返回的二值判断标签是否成立
            print(f"[{status}]{attr_name}: {prob:.6f}")
    except FileNotFoundError as e:
        print(f'异常:{e}')

# 使用CelebA中的图片
def test_model_by_celeba(model, test_loader, img_idx=0, sample_idx=0, threshold=0.5):
    # 从DataLoader中获取图片与标签
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    if sample_idx > 0:
        for _ in range(sample_idx):
            images, labels = next(test_iter)
    model.eval()
    with torch.no_grad():
        image_tensor = images[img_idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
        preds = probs > threshold
    print("预测结果:")
    for _, (attr_name, prob, pred) in enumerate(zip(celeba_classes, probs, preds)):
        status = "✓" if pred else "✗"
        print(f"[{status}]{attr_name}: {prob:.6f}")
    print("真实标签:")
    actual_labels = labels[img_idx].numpy()
    for _, (attr_name, actual_label) in enumerate(zip(celeba_classes, actual_labels)):
        status = "✓" if actual_label == 1 else "✗"
        print(f"[{status}]{attr_name}")

# 开始训练MLP与CNN
def begin_to_train():
    print('MLP训练:')
    iteratively_train(training_dataloader, test_dataloader, mlp_model, loss_func,
                      mlp_optim, mlp_sched, 'mlp.pth', 16)
    print('CNN训练:')
    iteratively_train(training_dataloader, test_dataloader, cnn_model, loss_func,
                      cnn_optim, cnn_sched, 'cnn.pth', 16)

def main():
    # begin_to_train()  # 如果还没训练, 先调用此函数
    load_model(mlp_model, 'mlp.pth')
    load_model(cnn_model, 'cnn.pth')
    # print('CelebA样本测试:')
    print('MLP结果:')
    # test_model_by_celeba(fnn_model, test_dataloader)
    use_model(mlp_model, 'faces/Lecun.jpg')
    print('CNN测试结果:')
    # test_model_by_celeba(cnn_model, test_dataloader)
    use_model(cnn_model, 'faces/Lecun.jpg')

main()