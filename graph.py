import torch
from models.LCCNet import LCCNet
from torchviz import make_dot, make_dot_from_trace
from tensorboardX import SummaryWriter
import netron


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


input_size = (256, 512)
model = LCCNet(input_size)
camera_input = torch.randn(1, 3, 256, 512).to(device)
lidar_input = torch.randn(1, 1, 256, 512).to(device)

netron.start('LCCnet.pt')
# torch.save(model, 'LCCnet.pt')

# with SummaryWriter("./log",comment="LCC") as sw:
#     sw.add_graph(model, [camera_input, lidar_input])

# output1, output2 = model(camera_input, lidar_input)

# with torch.no_grad():
#     torch.onnx.export(
#         model,
#         (camera_input, lidar_input),
#         'LCCNet.onnx',
#         verbose=True,
#         keep_initializers_as_inputs=True,
#     )


# a = make_dot(model(camera_input, lidar_input), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
# a.view()
# print(model)