%close all
source = pcread('test_filtered/test3.ply');
target = pcread('../../sofa/meshes/gel_phantom_1.ply');
block = stlread('../../sofa/meshes/gel_phantom_1_fine.stl');
test = pcread('../../dataset/2019-10-09-GelPhantom1/camera/data0/1570635131191.ply')
block_pts = pointCloud(block.Points);
t = [1000 0 0 0; 0 1000 0 0; 0 0 1000 0; 0 0 0 1]'*[eul2rotm([0 pi/2 pi/3], 'XYZ'), [0;0;0]; 0,0.15,0,1]'
scale_tform = affine3d(t');
source=  pctransform(source,scale_tform);

plot_test = pctransform(test,scale_tform);
figure
pcshow(source)
hold on
pcshow(plot_test)
pcshow(target)
pcshow(block_pts)
hold off

%figure
%pcshow(source)
source_denoised = source;%pcdenoise(source, 'threshold', 0.01);
%figure
%pcshow(source_denoised)
tform = pcregrigid(source_denoised,target,'Extrapolate',true);
tform.T'*scale_tform.T'
transform = affine3d((tform.T'*scale_tform.T')');
%test = pctransform(test,scale_tform);
registered = pctransform(test,transform);
figure
pcshow(registered)
hold on
pcshow(block_pts)
hold off
