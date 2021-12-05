import paddle
import paddle.nn as nn
from resnet18 import ResNet18, main
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter
import paddle.optimizer as optim
import paddle.metric as metric


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=20):
    print(f'----- Training Epoch [{epoch}/{total_epoch}]:')

    model.train()
    # 统计损失
    loss_meter = AverageMeter()
    # 统计准确率
    acc_meter = AverageMeter()

    for batch_idx, data in enumerate(dataloader):
        image = data[0]
        label = data[1]
        out = model(image)  # 前向传播
        loss = criterion(out, label)  # 计算损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.clear_grad()  # 清空梯度

        pred = nn.functional.softmax(out, axis=1)
        # unsqueeze(-1)将一维数据转换为二维数据，-1表示最后一维
        # accuracy的pred的shape为(batch_size, 10)，label的shape为(batch_size,1)
        acc = metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc.cpu().numpy()[0], batch_size)
        if batch_idx > 0 and batch_idx % report_freq == 0:
            print(f'----- Batch[{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')

    print(f'----- Epoch[{epoch}/{total_epoch}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')

def validate(model, dataloader, criterion, report_freq=10):
    print('----- Validation')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    for batch_idx, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)
        loss = criterion(out, label)

        pred = paddle.nn.functional.softmax(out, axis=1)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))
        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)

        if batch_idx > 0 and batch_idx % report_freq == 0:
            print(f'----- Batch [{batch_idx}/{len(dataloader)}], Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')

    print(f'----- Validation Loss: {loss_meter.avg:.5}, Acc@1: {acc_meter.avg:.4}')



def main():
    total_epoch = 200
    batch_size = 16  # 调成512，16为了方便调试

    model = ResNet18(num_classes=10)
    train_dataset = get_dataset(mode='train')
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, mode='train')
    val_dataset = get_dataset(mode='test')
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, mode='test')

    criterion = nn.CrossEntropyLoss()
    # 让lr使用余弦曲线的方式，让lr在训练的过程中不断的变化,200个epoch后，lr变成0
    # paddle.optimizer.lr学习率变化的函数
    scheduler = optim.lr.CosineAnnealingDecay(0.02, total_epoch)
    optimizer = optim.Momentum(learning_rate=scheduler,
                               parameters=model.parameters(),
                               momentum=0.9,
                               weight_decay=5e-4)

    for epoch in range(1, total_epoch+1):
        train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, total_epoch)
        scheduler.step()
        validation(model, val_dataloader, criterion)


if __name__ == '__main__':
    main()
