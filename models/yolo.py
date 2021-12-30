# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
# 获取当前文件路径的绝对路径
FILE = Path(__file__).resolve()
# 获取当前文件所在的文件夹路径
ROOT = FILE.parents[1]  # YOLOv5 root directory
# 如果Root未加入系统路径，则将该路径加入
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# 作用：将backbone中提取的特征，再次提取为位置信息:例如(n, 255, 20, 20) -> (n, 3, nc+5, 20, 20)
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        # 每一个 预选框预测输出，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5  # number of outputs per anchor
        # 表示预选层数，yolov5是3层预选
        self.nl = len(anchors)
        # 预选框数量，anchors数据中每一对数据表示一个预选框的宽高
        self.na = len(anchors[0]) // 2
        # 初始化grid列表大小，空列表
        self.grid = [torch.zeros(1)] * self.nl
        # 初始化anchor_grid列表大小，空列表
        self.anchor_grid = [torch.zeros(1)] * self.nl
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入 ---- 实际存的是框的宽高
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        # 每一张进行三次预测，每一个预测结果包含nc+5个值
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20) --> ch=(255, 255, 255)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        # 输入的x是来自三层金字塔的预测结果(n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20)
        for i in range(self.nl):
            # 下面3行代码的工作：
            # (n, 255, _, _) -> (n, 3, nc+5, ny, nx) -> (n, 3, ny, nx, nc+5)
            # 相当于三层分别预测了80*80、40*40、20*20次，每一次预测都包含3个框
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            # contiguous 将数据保证内存中位置连续
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # self.training 作为nn.Module的参数，默认是True，因此下方代码先不考虑
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 为每一层划分网格，gride是网格坐标，anchor_grid是预选框尺寸
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                # 改变原数据
                if self.inplace:
                    # grid[i] = (3, 20, 20, 2), y = [n, 3, 20, 20, nc+5]
                    # grid实际是 位置基准 或者理解为 cell的预测初始位置，而y[..., 0:2]是作为在grid坐标基础上的位置偏移
                    # anchor_grid实际是 预测框基准 或者理解为 预测框的初始位置，而 y[..., 2:4]是作为预测框位置的调整
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # stride应该是一个grid cell的实际尺寸
                    # 经过sigmoid，值范围变成了(0-1),下一行代码将值变成范围（-0.5，1.5），
                    # 相当于预选框上下左右都扩大了0.5倍的移动区域，不易大于0.5倍，否则就重复检验了其他网格的内容了
                    # 此处的1表示一个grid cell的尺寸，尽量让预测框的中心在grid cell中心附近
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # 范围变成(0-4)倍，设置为4倍的原因是下层的感受野是上层的2倍
                    # 因下层注重检测大目标，相对比上层而言，计算量更小，4倍是一个折中的选择
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            # 网格标尺坐标
            # indexing='ij' 表示的是i是同一行，j表示同一列
            # indexing='xy' 表示的是x是同一列，y表示同一行
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        # grid --> (20, 20, 2), 拓展（复制）成3倍，因为是三个框 -> (3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # 因为anchors在Model中被处理为了stride的倍数，也就是格子尺寸的倍数，这里再乘stride，就还原为了真实长度
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # 加载配置--字典方式和配置文件方式
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            # 获取文件路径中的文件名
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # 读取yaml文件, 得到数据字典
                self.yaml = yaml.safe_load(f)  # model dict

        # 如果配置文件没有规定 input channel，则使用传入的 input channel，并存入yaml字典
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        # 如果传入模型的 类别数 与配置文件的不一致，则优先使用传入值，并覆盖原值
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        # 如果传入了预选框，则优先使用传入值，并覆盖原值
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 拿到模型
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # 类别编号
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # 是否改变原数据
        self.inplace = self.yaml.get('inplace', True)

        # 获得检测模块
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 设定stride，骚操作通过结果的feature map尺寸来划分网格的cell大小，得到的是三层的cell size-->stride
            # stride 就是一格的尺寸， x.shape[-2] 是一个patch的尺寸
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 将anchor转换为 patch尺寸 的倍数，在Detect工作中，会被还原为真实长度
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查大网格是否对应大框框，小网格对应小框框，如果顺序反了则调整
            check_anchor_order(m)
            # 网格尺寸
            self.stride = m.stride
            # 初始化偏置
            self._initialize_biases()  # only run once
        # 初始化权重与偏置
        # Init weights, biases
        initialize_weights(self)
        # 模型规模评估
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile=profile, visualize=visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # 缩放
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            print("scale img")
            cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            # 判断是否有残差连接，-1表示单路径即没有残差连接
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            #
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            # 如果当前的m.i即编号，在残差列表中，则保留输出
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        # 如果是检测模块，则保存下来
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        # thop.profile计算 MACs, 表示乘加累计操作数，1MACs包含1个加和1个乘，约等于2个FLOPs，除以1E9将单位转换为G
        # FLOPS:表示每秒浮点运算次数
        # FLOPs:表示浮点运算的次数
        # MACs:表示乘加累计操作数
        # 如果是detect则拷贝一份作为输入
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        # 计算时间，在cuda开启状态会使用同步来精确计算
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # 每张图片尺寸 640 ，640/s表示以单元格衡量尺寸
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def print_biases(self):
        # 打印detect中将特征图映射为定位信息的卷积层中的偏置
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            # 打印前五个定位信息偏置，以及第六个类别偏置均值
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self
    # 模型规模评估
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


# 解析模型参数包，构建模型
def parse_model(d, ch):  # model_dict, input_channels(3)
    # 打印模型中各层参数--表头
    # LOGGER.info中 '':>3    表示靠右对齐，统一占位大于等于3
    #        'module':<40    表示靠左对齐，统一占位大于等于40
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # anchors：预选框
    # nc     ：类别数
    # depth_multiple： 网络深度防缩因子
    # width_multiple： 网络通道数防缩因子
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # 得到一层的预选框的个数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 每一个预选框包含 nc+5 个预测值，其中nc个值用来预测类别，2个用来预测位置调整，2个用来预测预测框尺寸调整，1个用来判别是否有目标
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # layers：保存每一层
    # save  ：为残差连接的concat暂存中间调整图
    # c2    ：当前模块的input_channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 构建模型
    # from：前一残差模块索引, number：深度, module：模块名, args：参数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # 实例化模块
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                # 将参数字符串转换为参数数值
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        # round(n * gd): 放缩模块深度，四舍五入
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 处理第一类模块，这一类模块的参数类似：
        # 输入、输出通道数、
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            # 每处计算c2目的是作为下一层的输入通道数
            c1, c2 = ch[f], args[0]
            # 判断是否为最终输出，否则输出通道数要经过放缩为模型量级
            if c2 != no:
                # 将c2限定下界为 8 ， make_divisible返回的是 max(ceil(c2 * gw / 8) * 8, 8)
                c2 = make_divisible(c2 * gw, 8)
            # 装入参数列表：输入通道数、输出通道数、其他参数
            args = [c1, c2, *args[1:]]
            # 处理第一类模块中的部分模块，这一类模块的参数：
            # 第三个参数额外多一个深度
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        # 处理第二类模块参数，仅仅需要一个通道数作为正则化参数
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # 处理第三类模块参数，计算出拼接后的通道数
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # 处理第四类模块参数，目标检测模块
        elif m is Detect:
            # 此模块参数为：[nc, anchors, [layer1_ch, layer2_ch, layer3_ch]]
            # 三层特征图的通道数
            args.append([ch[x] for x in f])
            # 如果anchors为整数，表明传入的是预选框个数，则生成anchors数对表示预选框,len(f)表示层数
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        # 处理第五类模块参数， 图像切割，例如（1，3，20，20）-->（1，12，10，10），传入参数为切割比例
        # 进而计算出输出的通道数为c2
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        # 处理第六类模块参数，是Contract的逆过程
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
        # 其他模块
            c2 = ch[f]
        # 逐个构建模块，对于多层的进行展开重构为整体，目的是提高效率，节省空间
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 例如："<class 'torch.nn.modules.conv.Conv2d'>"，目的为修改类型
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # numel()获取tensor的元素总个数，np就是总参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        # 修改模块的信息，便于后续制表格式打印
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 为残差连接的concat暂存中间调整图，同时也为可视化提供帮助
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        # 处理边界
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', default=False, action='store_true', help='profile model speed')
    parser.add_argument('--test', default=False, action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    # 检查文件是否是yaml类型的文件，不是则抛出异常
    opt.cfg = check_yaml(opt.cfg)
    # 打印： 文件名（不带后缀）、相关参数
    print_args(FILE.stem, opt)
    # 自动获得设备
    device = select_device(opt.device)

    # Create model
    # Model的构造器中为了获取网格数量，调用了一次forward，因此下面这一行代码就可以测试模型是否能运行
    model = Model(opt.cfg).to(device)
    # 设定为训练模式
    model.train()
    #model.print_biases()
    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        # profile：是否测算 FLOPs
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
