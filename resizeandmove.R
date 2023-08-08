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
