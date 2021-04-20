import scipy.spatial as sp
import numpy as np
import torch
import os
from colour import Color
# from Customer_Module.chamfer_distance.dist_chamfer import chamferDist
# from submodule.dist_chamfer_3D import chamfer_3DDist
from plyfile import PlyData, PlyElement
# nnd = chamferDist()
# nnd = chamfer_3DDist

outdir = "PCN"
# outdir = "MyResults"
colors = list(Color("blue").range_to(Color("red"),4))

def npy2ply(filename, save_filename):
    pts = np.load(filename)
    vertex = [tuple(item) for item in pts]
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(save_filename)

# def Eval_With_Charmfer_Distance():
#     print('Errors under Chamfer Distance')
#     for shape_id, shape_name in enumerate(shape_names):
#         gt_pts = np.load(os.path.join('./Dataset/Test', shape_name[:-6] + '.npy'))
#         pred_pts = np.load(os.path.join('./Dataset/Results', shape_name + '_pred_iter_2.npy'))
#         with torch.no_grad():
#             gt_pts_cuda = torch.from_numpy(np.expand_dims(gt_pts, axis=0)).cuda().float()
#             pred_pts_cuda = torch.from_numpy(np.expand_dims(pred_pts, axis=0)).cuda().float()
#             dist1, dist2 = nnd(pred_pts_cuda, gt_pts_cuda)
#             chamfer_errors = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
#             print('%12s  %.3f' % (models_name[shape_id], round(chamfer_errors.item() * 100000, 3)))

def Eval_With_Mean_Square_Error():
    print('Errors under Mean Square Error')
    for shape_id, shape_name in enumerate(shape_names):
        gt_pts = np.load(os.path.join('./Dataset/Test', shape_name[:-6] + '.npy'))
        gt_pts_tree = sp.cKDTree(gt_pts)
        # pred_pts = np.load(os.path.join('./Dataset/'+outdir, shape_name + '_pred_iter_2.npy'))
        pred_pts = np.loadtxt(os.path.join('./Dataset/' + outdir, shape_name + '_2.xyz'))
        pred_dist, _ = gt_pts_tree.query(pred_pts, 10)
        print('%12s  %.3f' % (models_name[shape_id], round(pred_dist.mean() * 1000, 3)))
        out = pred_dist.mean(1)*1000
        # print(out)
        color = match_color(out)
        # print(color)
        xyzrgb = np.hstack((pred_pts,color))
        # np.savetxt(os.path.join('./Dataset/'+outdir+"/eval", shape_name + '_pred_iter_2.txt'), xyzrgb, fmt='%.06f')
        np.savetxt(os.path.join('./Dataset/' + outdir + "/eval", shape_name + '_2.txt'), xyzrgb, fmt='%.06f')

# def File_Conversion():
#     for shape_id, shape_name in enumerate(shape_names):
#         npy2ply(os.path.join('./Dataset/'+outdir, shape_name + '_pred_iter_2.npy'),
#                 os.path.join('./Dataset/'+outdir, shape_name + '_pred_iter_2.ply'))

# def match_color(dist):
#     out = []
#     temp = None
#     for data in dist:
#         temp = colors[0]
#         for i in range(2000):
#             if data > i*0.01:
#                 temp = colors[i+1]
#         out.append(np.array(temp.get_rgb()))
#     return np.array(out)

def match_color(dist):
    out = []
    temp = None
    for data in dist:
        if data<5:
            temp = colors[0]
        elif data<15:
            temp = colors[1]
        elif data<25:
            temp = colors[2]
        else:
            temp = colors[3]
        out.append(np.array(temp.get_rgb()))
    return np.array(out)

if __name__ == '__main__':

    with open(os.path.join('./Dataset/Test', 'test.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    models_name = ['Boxunion',
                   'Cube',
                   'Fandisk',
                   'Tetrahedron']

    # File_Conversion()

    # Eval_With_Charmfer_Distance()
    Eval_With_Mean_Square_Error()
