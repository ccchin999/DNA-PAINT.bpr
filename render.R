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
