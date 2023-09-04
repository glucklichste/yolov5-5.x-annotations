## 从YOLOv5源码loss.py详细介绍Yolov5的损失函数

#### class ComputeLoss主要代码分析

1 __init__函数

代码和注释如下：

```python
class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
		
        # Define criteria
        '''
		定义分类损失和置信度损失为带sigmoid的二值交叉熵损失，
		即会先将输入进行sigmoid再计算BinaryCrossEntropyLoss(BCELoss)。
		pos_weight参数是正样本损失的权重参数。
		'''
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
		
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        '''
		对标签做平滑,eps=0就代表不做标签平滑,那么默认cp=1,cn=0
        后续对正类别赋值cp，负类别赋值cn
		'''
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        '''
		超参设置g>0则计算FocalLoss
		'''
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
		'''
		获取detect层
		'''
        m = de_parallel(model).model[-1]  # Detect() module
        '''
        每一层预测值所占的权重比，分别代表浅层到深层，小特征到大特征，4.0对应着P3，1.0对应P4,0.4对应P5。
        如果是自己设置的输出不是3层，则返回[4.0, 1.0, 0.25, 0.06, .02]，可对应1-5个输出层P3-P7的情况。
        '''
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        '''
        autobalance 默认为 False，yolov5中目前也没有使用 ssi = 0即可
        三个预测头的下采样率det.stride: [8, 16, 32].index(16): 求出下采样率stride=16的索引
        这个参数会用来自动计算更新3个feature map的置信度损失系数self.balance
        '''
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        '''
        赋值各种参数,gr是用来设置IoU的值在objectness loss中做标签的系数, 
        使用代码如下：
		tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
        train.py源码中model.gr=1，也就是说完全使用标签框与预测框的CIoU值来作为该预测框的objectness标签。
        self.BCEcls: 类别损失函数   self.BCEobj: 置信度损失函数   self.hyp: 超参数
        self.gr: 计算真实框的置信度标准的iou ratio    self.autobalance: 是否自动更新各feature map的置信度损失平衡系数  默认False
        '''
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors 3
        self.nc = m.nc  # number of classes 80 
        self.nl = m.nl  # number of layers 3
        self.anchors = m.anchors  # [[[1.25,1.625],[2.0,3.75],[4.125,2.875]],
                                    #  [[1.875, 3.8125],[3.875,2.8125][3.6875,7.4375]],
                                    #  [[3.625,2.8125],[4.875,6.1875],[11.65625,10.1875]]] 
        self.device = device
```

## 2 build_targets函数

代码和注释如下：

```python
    def build_targets(self, p, targets):
        """
        Build targets for compute_loss()
        na = 3,表示每个预测层anchors的个数
        nt为一个batch中所有标签的数量、如下面的targets中的63
        :params p: 预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                   tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                   如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                   [bs, anchor_num, grid_h, grid_w, xy+wh+confidence+classes] xy是相对grid_cell中心的偏移，wh是anchor尺寸基础上的偏差
                   可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
            targets 为一个batch中所有的标签，包括标签所属的image，以及class,x,y,w,h
            targets = [[image1,class1,x1,y1,w1,h1],
        		       [image2,class2,x2,y2,w2,h2],
        		       ...
        		       [imageN,classN,xN,yN,wN,hN]]
        :return tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                indices: b: 表示这个target属于的image index
                         a: 表示这个target使用的anchor index
                        gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                        gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors 3, targets 63
        tcls, tbox, indices, anch = [], [], [], [] # 初始化tcls tbox indices anch
        '''
        gain是为了后面将targets=[na,nt,7]中的归一化了的xywh映射到相对feature map尺度上（grid cell）,
        其中7是为了对应image_index+class+xywh+anchor_index: image class x y w h ai,
        但后续代码只对x y w h赋值，x,y,w,h = nx,ny,nx,ny,
        nx和ny为当前输出层的grid大小。
        '''
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        '''
        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 63]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        ai.shape = [na,nt]
        ai = [[0,0,0,.....],
        	  [1,1,1,...],
        	  [2,2,2,...]]
        这么做的目的是为了给targets增加一个属性，即当前标签所属的anchor索引
        '''
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''
        targets.repeat(na, 1, 1).shape = [na,nt,6]
        ai[:, :, None].shape = [na,nt,1](None在list中的作用就是在插入维度1)
        ai[:, :, None] = [[[0],[0],[0],.....],
        	  			  [[1],[1],[1],...],
        	  	  		  [[2],[2],[2],...]]
        cat之后：
        targets.shape = [na,nt,7]
        targets = [[[image1,class1,x1,y1,w1,h1,0],
        			[image2,class2,x2,y2,w2,h2,0],
        			...
        			[imageN,classN,xN,yN,wN,hN,0]],
        			[[image1,class1,x1,y1,w1,h1,1],
        			 [image2,class2,x2,y2,w2,h2,1],
        			...],
        			[[image1,class1,x1,y1,w1,h1,2],
        			 [image2,class2,x2,y2,w2,h2,2],
        			...]]
        这么做是为了纪录每个label对应的anchor。
        # [63, 6] [3, 63] -> [3, 63, 6] [3, 63, 1] -> [3, 63, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target对三个anchor都是正样本(复制三份) 再进行筛选  并将ai加进去标记当前是哪个anchor的target
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
		'''
		定义每个grid偏移量，会根据标签在grid中的相对位置来进行偏移
		这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
		'''
        g = 0.5  # bias   中心偏移  用来衡量target中心点离哪个格子更近
        '''
        以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        [0, 0]代表中间,
		[1, 0] * g = [0.5, 0]代表往左偏移半个grid， [0, 1]*0.5 = [0, 0.5]代表往上偏移半个grid，与后面代码的j,k对应
		[-1, 0] * g = [-0.5, 0]代代表往右偏移半个grid， [0, -1]*0.5 = [0, -0.5]代表往下偏移半个grid，与后面代码的l,m对应
		具体原理在代码后讲述
        '''
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets
		# 遍历三个feature map 筛选每个feature map(包含batch张图片)的每个anchor的正样本
        for i in range(self.nl): # self.nl: number of detection layers   Detect的个数 = 3
            '''
        	原本yaml中加载的anchors.shape = [3,6],但在yolo.py的Detect中已经通过代码
        	a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        	self.register_buffer('anchors', a) 
        	将anchors进行了reshape。
        	self.anchors.shape = [3,3,2]
        	anchors.shape = [3,2]
        	# anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
        	'''
            anchors, shape = self.anchors[i], p[i].shape
            '''
            p.shape = [nl,bs,na,nx,ny,no]
            p[i].shape = [bs,na,nx,ny,no]
            gain = [1,1,nx,ny,nx,ny,1]
            # gain: 保存每个输出feature map的宽高 -> gain[2:6]=gain[whwh]
            torch.tensor(shape)--> [63, 3, 80, 80, 85]
            torch.tensor(shape)[[3, 2, 3, 2]] --> [80, 80, 80, 80]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]] gain-->[1, 1, 80, 80, 80, 80, 1]
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 80, 80, 80, 80, 1]=image_index+class+xywh+anchor_index
            '''
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  #xyxy gain

            # Match targets to anchors
            '''
            因为targets进行了归一化，默认在w = 1, h =1 的坐标系中，
            需要将其映射到当前输出层w = nx, h = ny的坐标系中。
            '''
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                '''
                t[:, :, 4:6].shape = [na,nt,2] = [3,nt,2],存放的是标签的w和h
                anchor[:,None] = [3,1,2]
                r.shape = [3,nt,2],存放的是标签和当前层anchor的长宽比
                '''
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                 '''
                torch.max(r, 1. / r)求出最大的宽比和最大的长比，shape = [3,nt,2]
                再max(2)求出同一标签中宽比和长比较大的一个，shape = [2，3,nt],之所以第一个维度变成2，
                因为torch.max如果不是比较两个tensor的大小，而是比较1个tensor某一维度的大小，则会返回values和indices：
                	torch.return_types.max(
						values=tensor([...]),
						indices=tensor([...]))
                所以还需要加上索引0获取values，
                torch.max(r, 1. / r).max(2)[0].shape = [3,nt],
                将其和hyp.yaml中的anchor_t超参比较，小于该值则认为标签属于当前输出层的anchor
                j = [[bool,bool,....],[bool,bool,...],[bool,bool,...]]
                j.shape = [3,nt]
                '''
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                '''
                t.shape = [na,nt,7] 
                j.shape = [3,nt]
                假设j中有NTrue个True值，则
                t[j].shape = [NTrue,7]
                返回的是na*nt的标签中，所有属于当前层anchor的标签。
                '''
                t = t[j]  # filter

                # Offsets
                '''
                下面这段代码和注释可以配合代码后的图片进行理解。
                t.shape = [NTrue,7] 
                7:image,class,x,y,h,w,ai
                gxy.shape = [NTrue,2] 存放的是x,y,相当于坐标到坐标系左边框和上边框的记录
                gxi.shape = [NTrue,2] 存放的是w-x,h-y,相当于测量坐标到坐标系右边框和下边框的距离
                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                '''
                gxy = t[:, 2:4]  # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                gxi = gain[[2, 3]] - gxy  # inverse
                '''
                因为grid单位为1，共nx*ny个gird
                gxy % 1相当于求得标签在第gxy.long()个grid中以grid左上角为原点的相对坐标，
                gxi % 1相当于求得标签在第gxy.long()个grid中以grid右下角为原点的相对坐标，
                下面这两行代码作用在于
                筛选中心坐标 左、上方偏移量小于0.5,并且中心点大于1的标签
                筛选中心坐标 右、下方偏移量小于0.5,并且中心点大于1的标签          
                j.shape = [NTrue], j = [bool,bool,...]
                k.shape = [NTrue], k = [bool,bool,...]
                l.shape = [NTrue], l = [bool,bool,...]
                m.shape = [NTrue], m = [bool,bool,...]
                '''
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                '''
                j.shape = [5,NTrue]
                t.repeat之后shape为[5,NTrue,7], 
                通过索引j后t.shape = [NOff,7],NOff表示NTrue + (j,k,l,m中True的总数量)
                torch.zeros_like(gxy)[None].shape = [1,NTrue,2]
                off[:, None].shape = [5,1,2]
                相加之和shape = [5,NTrue,2]
                通过索引j后offsets.shape = [NOff,2]
                这段代码的表示当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，上下右同理。
                '''
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            '''
            t.shape = [NOff,7],(image,class,x,y,w,h,ai)
            b, c：image_index, class
            gxy：target的xy
            gwh：target的wh
            '''
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            '''
            offsets.shape = [NOff,2]
            gxy - offsets为gxy偏移后的坐标，
            gxi通过long()得到偏移后坐标所在的grid坐标
            '''
            gij = (gxy - offsets).long() # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
            gi, gj = gij.T  # grid xy indices

            # Append
            '''
            a:所有anchor的索引 shape = [NOff]
            b:标签所属image的索引 shape = [NOff]
            gj.clamp_(0, gain[3] - 1)将标签所在grid的y限定在0到ny-1之间
            gi.clamp_(0, gain[2] - 1)将标签所在grid的x限定在0到nx-1之间
            indices = [image, anchor, gridy, gridx] 最终shape = [nl,4,NOff]
            tbox存放的是标签在所在grid内的相对坐标，∈[0,1] 最终shape = [nl,NOff]
            anch存放的是anchors 最终shape = [nl,NOff,2]
            tcls存放的是标签的分类 最终shape = [nl,NOff]
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            '''
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # tbix: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # 对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
```

在上述论文中的代码中包含了标签偏移的代码部分：

```python
   g = 0.5  # bias
   off = torch.tensor([[0, 0],
                  [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                  # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                  ], device=targets.device).float() * g  # offsets
  # Offsets
  gxy = t[:, 2:4]  # grid xy
  gxi = gain[[2, 3]] - gxy  # inverse
  j, k = ((gxy % 1. < g) & (gxy > 1.)).T
  l, m = ((gxi % 1. < g) & (gxi > 1.)).T                    
  j = torch.stack((torch.ones_like(j), j, k, l, m))
  t = t.repeat((5, 1, 1))[j]
  offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
  gxy = t[:, 2:4]  # grid xy
  gij = (gxy - offsets).long()
```

在讲述yolo.py的时候也已经介绍过，这里再介绍一遍。
这段代码的大致意思是，当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，在上、下、右侧同理。具体如图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/3a2383bcc9514232bd82ca29a4620c9d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASG9sbG93S25pZ2h0Wg==,size_9,color_FFFFFF,t_70,g_se,x_16)

grid B中的标签在右上半部分，所以标签偏移0.5个gird到E中，A,B,C,D同理，即每个网格除了回归中心点在该网格的目标，还会回归中心点在该网格附近周围网格的目标。以E左上角为坐标$（C_x,C_y）$，所以b$x∈[C_x-0.5,C_x+1.5]，b_y∈[C_y-0.5,C_y+1.5]$，而$b_w∈[0,4p_w]，b_h∈[0,4p_h]$应该是为了限制anchor的大小。


## 3 _call__函数

代码和注释如下：

```python
    def __call__(self, p, targets):  # predictions, targets
        """
        :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                    [bs, anchor_num, grid_h, grid_w, xywh+ confidence +classes]
                    可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_object,  batch_index+class+xywh]
        targets 为一个batch中所有的标签，包括标签所属的image，以及class,x,y,w,h
        targets = [[image1,class1,x1,y1,w1,h1],
        		   [image2,class2,x2,y2,w2,h2],
        		   ...
        		   [imageN,classN,xN,yN,wN,hN]]
        :params loss * bs: 整个batch的总损失  进行反向传播
        :params torch.cat((lbox, lobj, lcls, loss)).detach(): 回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        
        # 初始化lcls, lbox, lobj三种损失值  tensor([0.])
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        '''
        每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        tcls: 表示这个target所属的class index tcls = [[cls1,cls2,...],[cls1,cls2,...],[cls1,cls2,...]]  tcls.shape = [nl,N]
        tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量 tbox = [[[gx1,gy1,gw1,gh1],[gx2,gy2,gw2,gh2],...],
        indices: b: 表示这个target属于的image index
                 a: 表示这个target使用的anchor index
                 gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                 gi: 表示这个网格的左上角x坐标
                 indices = [[image indices1,anchor indices1,gridj1,gridi1],
        		            [image indices2,anchor indices2,gridj2,gridi2],
        		            ...]]
        anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算 anchors = [[aw1,ah1],[aw2,ah2],...]	
        从build_targets函数中构建目标标签，获取标签中的tcls, tbox, indices, anchors     	  
        '''
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        '''
		p.shape = [nl,bs,na,nx,ny,no]
		nl 为 预测层数，一般为3
		na 为 每层预测层的anchor数，一般为3
		nx,ny 为 grid的w和h
		no 为 输出数，为5 + nc (5:x,y,w,h,obj,nc:分类数)
		'''
        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            a:所有anchor的索引
            b:标签所属image的索引
            gridy:标签所在grid的y，在0到ny-1之间
            gridy:标签所在grid的x，在0到nx-1之间
            '''
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            '''
            pi.shape = [bs,na,nx,ny,no]
            tobj.shape = [bs,na,nx,ny]
            '''
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                '''
                只有当CIOU=True时，才计算CIOU，否则默认为GIOU
                '''
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                '''
                通过gr用来设置IoU的值在objectness loss中做标签的比重, 
                '''
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    '''
               		ps[:, 5:].shape = [N,nc],用 self.cn 来填充型为[N,nc]得Tensor。
               		self.cn通过smooth_BCE平滑标签得到的，使得负样本不再是0，而是0.5 * eps
                	'''
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    '''
                    self.cp 是通过smooth_BCE平滑标签得到的，使得正样本不再是1，而是1.0 - 0.5 * eps
                    '''
                    t[range(n), tcls[i]] = self.cp
                    '''
                    计算用sigmoid+BCE分类损失
                    '''
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
			'''
			pi[..., 4]所存储的是预测的obj
			'''
            obji = self.BCEobj(pi[..., 4], tobj)
            '''
			self.balance[i]为第i层输出层所占的权重，在init函数中已介绍
			将每层的损失乘上权重计算得到obj损失
			'''
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        '''
        hyp.yaml中设置了每种损失所占比重，分别对应相乘
        '''
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
```

在anchor回归时，对xywh进行了以下处理：

```python
 # Regression
 pxy = ps[:, :2].sigmoid() * 2. - 0.5
 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
```

这和yolo.py Detect中的代码一致：

```python
y = x[i].sigmoid()
y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
```

可以先翻看yolov3论文中对于anchor box回归的介绍：

![在这里插入图片描述](https://img-blog.csdnimg.cn/58358b5150c2425091200ca12c70e623.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASG9sbG93S25pZ2h0Wg==,size_12,color_FFFFFF,t_70,g_se,x_16)

这里的$b_x∈[C_x,C_x+1]，b_y∈[C_y,C_y+1]，b_w∈(0，+∞)，b_h∈(0，+∞)$
而yolov5里这段公式变成了：

![在这里插入图片描述](https://img-blog.csdnimg.cn/c4c2b76686664511ab06fee484d4a892.png)

使得$b_x∈[C_x-0.5,C_x+1.5]，b_y∈[C_y-0.5,C_y+1.5]，b_w∈[0,4p_w]，b_h∈[0,4p_h]$。
这么做是为了消除网格敏感，因为sigmoid函数想取到0或1需要趋于正负无穷，这对网络训练来说是比较难取到的，所以通过往外扩大半个格子范围，是的格点边缘上的点也能取到。
这一策略可以提高召回率（因为每个grid的预测范围变大了），但会略微降低精确度，总体提升mAP。
原文链接：https://blog.csdn.net/Z960515/article/details/122357364