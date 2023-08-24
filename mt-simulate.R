args=commandArgs(TRUE)
sdense=as.numeric(args[1])
sp=as.numeric(args[2])
dsize=as.numeric(args[3])
opat=args[4]
print("Please give structure_dense, dataset_size and output_dataset_path.")

if(!require("EBImage")){
  if(!require("BiocManager")){
    install.packages("BiocManager")
    require("BiocManager")
  }
  BiocManager::install("EBImage")
  require("EBImage")
}

mtSimulate=function(dense=5){
  mtNum=ceiling(dense+rnorm(1)+runif(1,-1*dense*0.5,dense*0.5))
  interval=0.005
  count=500
  sd=interval/50
  d=0
  for(i in 1:mtNum){
    d=c(d,rep(i,count))
  }
  d=d[-1]
  d=data.frame(x=0,y=0,mt=d)
  for(i in 1:mtNum){
    a=runif(1,0,2*pi)
    d$x[d$mt==i][1]=sqrt(2)*cos(a)
    d$y[d$mt==i][1]=sqrt(2)*sin(a)
    # a=runif(1,0,2*pi)
    d$x[d$mt==i][2]=cos(a)*(sqrt(2)-interval)
    d$y[d$mt==i][2]=sin(a)*(sqrt(2)-interval)
    for(c in 3:count){
      d$x[d$mt==i][c]=2*d$x[d$mt==i][c-1]-d$x[d$mt==i][c-2]
      d$y[d$mt==i][c]=2*d$y[d$mt==i][c-1]-d$y[d$mt==i][c-2]
      d$x[d$mt==i][c]=d$x[d$mt==i][c]+rnorm(1,0,sd)
      d$y[d$mt==i][c]=d$y[d$mt==i][c]+rnorm(1,0,sd)
    }
  }
  # d$x=d$x-min(d$x)
  # d$y=d$y-min(d$y)
  # d$x=d$x/max(d$x)
  # d$y=d$y/max(d$y)
  # plot(d$x[-1<=d$x & d$x <=1 & -1<=d$y & d$y<=1],d$y[-1<=d$x & d$x <=1 & -1<=d$y & d$y<=1],pch='.',asp=1)
  # plot(d$x,d$y,pch='.',asp=1)
  # lines(sqrt(2)*cos(0:100/100*2*pi),sqrt(2)*sin(0:100/100*2*pi),col='red')
  # lines(c(1,1,-1,-1,1),c(1,-1,-1,1,1),col='blue')
  data.frame(x=d$x[-1<=d$x & d$x <=1 & -1<=d$y & d$y<=1],y=d$y[-1<=d$x & d$x <=1 & -1<=d$y & d$y<=1])
}

zeromax=function(im=0){
  m=max(im)
  if(m==0)
    m+1
  else
    m
}

render=function(x,y,w=1024,h=1024){
  if(length(x)==length(y) && w>0 && h>0){
    x=x-min(x)+10^(-16)
    y=y-min(y)+10^(-16)
    x=x*(w/zeromax(x))
    y=y*(h/zeromax(y))
    im=matrix(0,ceiling(w),ceiling(h))
    for(i in 1:length(x)){
      im[ceiling(x[i]),ceiling(y[i])]=im[ceiling(x[i]),ceiling(y[i])]+1
    }
    im=im/zeromax(im)
    # EBImage::display(im,method = 'raster')
    im
  }
  else
    print("x and y length differs.")
}

resize=function(image,w=0,h=0){
  if(w<=0){
    if(h<=0){
      stop("either 'w' or 'h' must be specified")
    }else{
      w=round(nrow(image)*h/ncol(image))
    }
  }else{
    if(h<=0){
      h=round(ncol(image)*w/nrow(image))
    }
  }
  wi=nrow(image)/w
  hi=ncol(image)/h
  image[ceiling((1:w)*wi),ceiling((1:h)*hi)]
  #    im=matrix(0,w,h)
  #    for(i in 1:w){
  #        for(j in 1:h){
  #            
  #        }
  #    }
}

move=function(image,dw=0,dh=0){
  dw=round(-dw)
  dh=round(-dh)
  w=dim(image)[1]
  h=dim(image)[2]
  if(dh!=0)
    for(i in 1:abs(dh))
      image=cbind(0,image,0)
  if(dw!=0)
    for(i in 1:abs(dw))
      image=rbind(0,image,0)
  if(dw>0)
    dw=0
  else
    dw=2*dw
  if(dh>0)
    dh=0
  else
    dh=2*dh
  image[(1:w)-dw,(1:h)-dh]
}

dir.create(opat,showWarnings = FALSE, recursive = TRUE)
setwd(opat)
gb=160/128
for(i in 1:dsize){
  if(dir.exists(as.character(i))){
    print(paste("Simulated data",i,"has been created before. Stop simulation."))
    break()
  }
  dir.create(as.character(i))
  print(paste("Processing",i,"."))
  d=mtSimulate(runif(1,sdense/2,sdense*1.5))
  # im=render(d$x,d$y,256,256)
  # EBImage::writeImage(im,file.path(i,"micro-tubules.tif"))
  sd=0.003
  im=render(d$x+rnorm(length(d$x),0,sd),d$y+rnorm(length(d$y),0,sd),256,256)
  im=EBImage::gblur(im,gb)
  EBImage::writeImage(im,file.path(i,"ground-truth.tif"))
  im=EBImage::resize(im,16)
  # EBImage::writeImage(resize(im,256),file.path(i,"WF.tif"))
  im=EBImage::resize(im,256)
  EBImage::writeImage(im,file.path(i,"wide-field.tif"))
  # Sparsely localized image
  a=round(runif(length(d$x)*sp,1,length(d$x)))
  d=data.frame(x=d$x[a],y=d$y[a])
  im=render(d$x+rnorm(length(d$x),0,sd),d$y+rnorm(length(d$y),0,sd),256,256)
  EBImage::writeImage(EBImage::gblur(im,gb),file.path(i,"sparsely-localized.tif"))
}
