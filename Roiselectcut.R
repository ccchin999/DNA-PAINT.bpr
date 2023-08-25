args=commandArgs(TRUE)
tiffpat=args[1]
opat=args[2]
size=as.numeric(args[3])
print("Please give uncropped_file_path, output_dataset_path and target_image_size.")

# pat,grid_x
Roiselectcut=function(pat,
outpat=pat,
grid_x=256,
start_x=1,
start_y=1,
w=0,
h=0,
end_x=w-grid_x+1,
end_y=h-grid_y+1,
grid_y=grid_x,
roipath=file.path(pat,paste0('RoiSelectOf',grid_x,'x',grid_y,'.csv')),
startcount=1){
    nam=tail(strsplit(pat,.Platform$file.sep)[[1]],1)[1]
    tif=dir(pat,'*.tif')
    im=EBImage::readImage(file.path(pat,tif[1]))
    if(w == 0)
        w=nrow(im)
    if(h == 0)
        h=ncol(im)
    if(end_x <= 0)
        end_x=w-grid_x+1
    if(end_y <=0)
        end_y=h-grid_y+1
    im=array(0,c(w,h,length(tif)))
    for(i in 1:length(tif)){
        im[,,i]=EBImage::resize(EBImage::readImage(file.path(pat,tif[i])),w=w,h=h)
    }
    pmean=1.2*mean(im[,,1])
    pmax=0.8*max(im[,,1])
    roi=data.frame(x1=0,x2=0,y1=0,y2=0)
    for(i_x in seq(start_x,end_x,grid_x)){
        for(i_y in seq(start_y,end_y,grid_y)){
            tim=im[i_x:(i_x+grid_x-1),i_y:(i_y+grid_y-1),1]
            mmean=mean(tim)
            mmedian=median(tim)
            if(mmean>=pmean && mmean<=pmax && 0.8*mmean>mmedian && mmedian>= 0.1*mmean){
                x1=i_x
                x2=(i_x+grid_x-1)
                y1=i_y
                y2=i_y+grid_y-1
                roi=rbind(roi,c(x1,x2,y1,y2))
            }
        }
    }
    roi=roi[-1,]
    write.table(roi,roipath,sep=',',row.names=FALSE)
    
    for(i in 1:length(roi$x1)){
        outp=file.path(outpat,as.character(startcount+i-1))
        dir.create(outp,recursive = TRUE)
        for(f in 1:length(tif)){
            EBImage::writeImage(im[roi$x1[i]:roi$x2[i],roi$y1[i]:roi$y2[i],f],file.path(outp,tif[f]))
        }
        cat(paste0("Tiff files from ",pat,". Roi index is ",i,". Cropped area is x: [",roi$x1[i],",",roi$x2[i],"] y: [",roi$y1[i],",",roi$y2[i],"]."),file=file.path(outp,paste0(nam,"-statements.txt")))
    }
    startcount+i-1
}

Roiselectcut(pat=tiffpat,outpat=opat,grid_x=size,grid_y=size,start_x=size/4+1,start_y=size/4+1)
