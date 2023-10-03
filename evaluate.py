from multiprocessing.spawn import get_preparation_data
import open3d
import os
import numpy
from scipy.spatial.transform import Rotation as R


def LoadTraj(format, pathToTxt):
    trajPoses = []
    if format == 'tum':
        with open(pathToTxt, 'r') as trajFile:
            trajListStr = trajFile.readlines()
        for line in trajListStr:
            if line[-1] == "\n":
                line = line[:-1]
            sLine = line.split(" ")
            if len(sLine) == 1:
                sLine = line.split(",")
            r = R.from_quat([float(sLine[-1]), float(sLine[-4]), float(sLine[-3]), float(sLine[-2])])
            rmatrix = r.as_matrix()
            onePose = numpy.eye(4)
            onePose[:3, :3] = rmatrix
            onePose[:3, 3] = numpy.array([
                float(sLine[1]),
                float(sLine[2]),
                float(sLine[3])
            ])
            trajPoses.append(onePose)
    else:
        raise NotImplementedError

    return trajPoses

def PrepareTrajAndGt(vTraj, vGt, vMask):
    if isinstance(vTraj, list):
        vTraj = numpy.concatenate(vTraj, axis=0)
    if isinstance(vGt, list):
        vGt = numpy.concatenate(vGt, axis=0)
    if vTraj.shape[0] > vGt.shape[0]:
        vTraj = vTraj[:vGt.shape[0]]
        if vMask is not None:
            vMask = vMask[:vGt.shape[0]]  # vMask always has same shape as vGt
    if vGt.shape[0] > vTraj.shape[0]:
        vGt = vGt[:vTraj.shape[0]]
    if vMask is not None:
        maskedTraj = []
        maskedGt = []
        for indexP, maskOneP in enumerate(vMask):
            if maskOneP:
                maskedTraj.append(vTraj[indexP][numpy.newaxis, :])
                maskedGt.append(vGt[indexP][numpy.newaxis, :])
        vTraj = numpy.concatenate(maskedTraj, axis=0)
        vGt = numpy.concatenate(maskedGt, axis=0)
    return vTraj, vGt

def ComputeRMSATE(points, gtpoints, mode):
    print("====================")
    if mode == '3d':
        pass
    elif mode == '2d':
        points[:, 2] = gtpoints[:, 2]

    errors = numpy.linalg.norm(points - gtpoints, axis=1)
    minError = numpy.min(errors)
    maxError = numpy.max(errors)
    meanError = numpy.mean(errors)
    rmsate = numpy.sqrt(numpy.sum(numpy.power(errors, 2)) / points.shape[0])
    if mode == '3d':
        print("RMS-ATE (m) between gt and tracked traj. in 3d is:\n     ", rmsate)
    elif mode == '2d':
        print("RMS-ATE between gt and tracked traj. in 2d is:\n     ", rmsate)
    print("min translation error (m):  ", minError)
    print("max translation error (m):  ", maxError)
    print("mean translation error (m):  ", meanError)
    print("====================")


# ======================== visualize ============================
def VisualizeTrajCompare(points, gtpoints):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(numpy.tile(numpy.array([1, 0, 0]), (points.shape[0], 1)))
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(gtpoints)
    pcd2.colors = open3d.utility.Vector3dVector(numpy.tile(numpy.array([0, 0, 1]), (gtpoints.shape[0], 1)))
    return pcd, pcd2


def VisualizeCorrespondence(points, gtpoints):
    pointsConcat = numpy.concatenate([points, gtpoints], axis=0)
    indiciesPoints = numpy.arange(0, points.shape[0])[:, numpy.newaxis]
    indiciesPointsGt = (numpy.arange(0, points.shape[0]) + points.shape[0])[:, numpy.newaxis]
    lines = numpy.hstack([indiciesPoints, indiciesPointsGt])
    colors = [[0, 1, 0] for i in range(lines.shape[0])]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(pointsConcat),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


if __name__ == "__main__":
    import argparse
    
    scriptName = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="Evaluate result of SLAM traj.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples: \n" +
                '%s --dataroot .../mydataset/ \n' % scriptName
        )
    )

    parser.add_argument('--dataroot', '-i', action='store', type=str, dest='dataroot',
                        help='Path to the data root.')
    parser.add_argument('--gtfilename', action='store', type=str, dest='gtfilename', default='gtpose_timealigned.txt',
                        help='gt file name.')
    parser.add_argument('--trajfilename', action='store', type=str, dest='trajfilename', default='cameraTrack.txt',
                        help='traj. file name.')
    parser.add_argument('--maskfilename', action='store', type=str, dest='maskfilename', default='none',
                        help='traj. mask file name.')
    parser.add_argument('--show', '-s', action='store_true', dest='show',
                        help='whether show visz.')

    args, remaining = parser.parse_known_args()

    gtPoses = LoadTraj('tum', os.path.join(args.dataroot, args.gtfilename))
    trackedPoses = LoadTraj('tum', os.path.join(args.dataroot, args.trajfilename))

    # get mask file
    if args.maskfilename != 'none':
        with open(os.path.join(args.dataroot, args.maskfilename), 'r') as maskFile:
            maskListStr = maskFile.readlines()
        maskList = []
        for oneEntry in maskListStr:
            if oneEntry != '\n':
                maskList.append(int(oneEntry))
        vMask = numpy.array(maskList).astype('bool')
    else:
        vMask = None

    gtpoints = []
    for gtPose in gtPoses:
        gtpoints.append(gtPose[:3, 3][numpy.newaxis, :])
    trackpoints = []
    for trackedPose in trackedPoses:
        trackpoints.append(trackedPose[:3, 3][numpy.newaxis, :])

    trackpoints, gtpoints = PrepareTrajAndGt(trackpoints, gtpoints, vMask)
    pcd, pcd2 = VisualizeTrajCompare(trackpoints, gtpoints)
    line_set = VisualizeCorrespondence(trackpoints, gtpoints)

    if args.show:
        open3d.visualization.draw_geometries([pcd, pcd2, line_set])
    else:
        ComputeRMSATE(trackpoints, gtpoints, mode='2d')

    numpy.save("traj.npy", trackpoints)
    numpy.save("gt.npy", gtpoints)
