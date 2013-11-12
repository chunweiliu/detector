function vis_image(id,flip,params)
VOCinit;

im = imread(sprintf(VOCopts.imgpath,id));
if flip
    im = flipdim(im,2);
end
load(sprintf(VOCopts.hogdatapath,id,flip,params.w,params.h,params.offset))

figure(1)
plot_boxes(im, posboxes, 'g');

figure(2)
plot_boxes(im, negboxes2, 'r');