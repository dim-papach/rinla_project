# Get data from a FITS file
get_data <- function(file_path) {
  # Read the FITS file
  fits_data <- readFITS(file_path)
  
  # Extract the image data from the FITS file
  image_data <- fits_data$imDat
  
  return(image_data)
}

#' Prepare Data for INLA Analysis
#'
#' This function prepares image data for INLA analysis by extracting dimensions, 
#' creating coordinate matrices, identifying valid data points, and normalizing the image data.
#'
#' @param img A numeric matrix representing the image data to be analyzed.
#'
#' @return A list containing the following elements:
#' \item{x}{A matrix of x-coordinates corresponding to the image data.}
#' \item{y}{A matrix of y-coordinates corresponding to the image data.}
#' \item{valid}{A vector of indices indicating valid (non-NA) data points in the image.}
#' \item{xsize}{The number of columns in the image data.}
#' \item{ysize}{The number of rows in the image data.}
#' \item{xfin}{The final x-dimension size, same as xsize.}
#' \item{yfin}{The final y-dimension size, same as ysize.}
#' \item{logimg}{A matrix of the log10-transformed image data.}
#'
#' @examples
#' img <- matrix(runif(100), nrow = 10, ncol = 10)
#' prepared_data <- prepare_data(img)
prepare_data <- function(img) {
    # Get dimensions of the img array
    dims <- dim(img)

    # Create x and y arrays using matrix indexing
    x <- matrix(rep(1:dims[1], dims[2]), nrow = dims[1], ncol = dims[2])
    y <- matrix(rep(1:dims[2], each = dims[1]), nrow = dims[1], ncol = dims[2])

    # Identify valid data points
    valid <- which(!(is.na(img)))

    # Set dimensions
    xsize <- dims[2]
    ysize <- dims[1]
    xfin <- xsize
    yfin <- ysize

    ##normalize data
    logimg = log10(img)

    return(list(x = x, y = y, valid = valid, xsize = xsize, ysize = ysize, xfin = xfin, yfin = yfin, logimg = logimg))
}


#' Perform INLA Analysis on Prepared Data
#'
#' This function performs INLA) analysis on prepared image data.
#' It supports both stationary and non-stationary models and can handle different shapes for the analysis.
#'
#' @param prepared_data A list containing prepared image data, including coordinates, valid data points, and log-transformed image data.
#' @param weight A numeric value representing the weight for the analysis. Default is 1.
#' @param zoom A numeric value representing the zoom factor for the analysis. Default is 1.
#' @param xini A numeric value representing the initial x-coordinate for the analysis. Default is 0.
#' @param yini A numeric value representing the initial y-coordinate for the analysis. Default is 0.
#' @param nonstationary A logical value indicating whether to use a non-stationary model. Default is FALSE.
#' @param restart An integer value representing the number of restarts for the INLA algorithm. Default is 0L.
#' @param shape A character string representing the shape for the analysis. Options are 'ellipse', 'radius', or 'none'. Default is 'ellipse'.
#' @param tolerance A numeric value representing the tolerance for the INLA algorithm. Default is 1e-4.
#' @param p_range A numeric vector representing the prior range for the Gaussian process. Default is c(2, 0.2).
#' @param p_sigma A numeric vector representing the prior sigma for the Gaussian process. Default is c(2, 0.2).
#' @param cutoff A numeric value representing the cutoff for the mesh tessellation. Default is 5.
#'
#' @return A list containing the following elements:
#' \item{out}{A matrix of the INLA analysis results.}
#' \item{image}{A matrix of the original image data.}
#' \item{erimage}{A matrix of the error image data, if available.}
#' \item{outsd}{A matrix of the standard deviation of the INLA analysis results.}
#' \item{x}{A vector of x-coordinates for the analysis results.}
#' \item{y}{A vector of y-coordinates for the analysis results.}
#' \item{z}{A vector of the INLA analysis results.}
#' \item{erz}{A vector of the standard deviation of the INLA analysis results.}
#'
stationary_inla <- function(prepared_data, weight=1, zoom = 1,
                            xini=0,yini=0,
                            nonstationary=FALSE,restart=0L,
                            shape='ellipse',tolerance=1e-4,
                            p_range=c(2,0.2),p_sigma=c(2,0.2),cutoff=5){

    # Extract variables
    valid <- prepared_data$valid
    tx <- prepared_data$x
    ty <- prepared_data$y
    xsize <- prepared_data$xsize
    ysize <- prepared_data$ysize
    xfin <- prepared_data$xfin
    yfin <- prepared_data$yfin
    logimg <- prepared_data$logimg

    x <- tx[valid]
    y <- ty[valid]

    par <- logimg[valid]
    if (hasArg(tepar)) { epar <- tepar^2 } else { epar <- NULL}

    # Create a mesh (tesselation) 
    mesh <- inla.mesh.2d(cbind(x,y), max.n = 10, cutoff = cutoff)

    #bookkeeeping
    A <- inla.spde.make.A(mesh, loc=cbind(x,y))

    #calculate projection from model
    projection <- inla.mesh.projector(mesh,xlim=c(xini,xfin),ylim=c(yini,yfin),
                                    dim=zoom*c(xsize+1,ysize+1))
        
    if (nonstationary){    

    #inverse scale: degree=10, degree=10, n=2, n=2
    basis.T <- inla.mesh.basis(mesh,type="b.spline", n=nbasis, degree=degree)
    #inverse range
    basis.K <- inla.mesh.basis(mesh,type="b.spline", n=nbasis, degree=degree)

    spde <- inla.spde2.matern(mesh=mesh, alpha=2,
                                B.tau=cbind(0,basis.T,basis.K*0),B.kappa=cbind(0,basis.T*0,basis.K/2))
    }
    else {
        
    #priors gaussian process (prior range: 20% less than 2, sigma 20 higher than 2)
    spde <- inla.spde2.pcmatern(mesh=mesh,alpha=2, prior.range=p_range,prior.sigma=p_sigma) 
    }
    
        
    #center
    xcenter <- sum(x*weight)/sum(weight)
    ycenter <- sum(y*weight)/sum(weight)
    #print(xcenter)
    #print(ycenter)
        
    #radius fct
    if(shape == 'radius'){
        radius <- sqrt((x-xcenter)^2 + (y-ycenter)^2)
        radius_2 <- (x-xcenter)^2 + (y-ycenter)^2

        #use parametric function of  radius+radius^2
        stk_rad <- inla.stack(data=list(par=par), A=list(A,1,1,1),
                            effects=list(i=1:spde$n.spde, m=rep(1,length(x)),
                                        radius=radius, radius_2=radius_2),tag='est')
    
        #caculate result model
        res <- inla(par ~ 0 + m +radius +radius_2 +f(i, model=spde),
                    data=inla.stack.data(stk_rad),
                    control.predictor=list(A=inla.stack.A(stk_rad)),scale=epar,
                    control.compute = list(openmp.strategy='huge'),
            control.inla = list(tolerance=tolerance,restart=restart))

        #porjection for radius
        projected_radius <- sqrt((rep(projection$x,each=length(projection$y))-xcenter)^2 +
                                (rep(projection$y,length(projection$x)) -ycenter)^2)
        projected_radius_2  <- (rep(projection$x,each=length(projection$y))-xcenter)^2 +
            (rep(projection$y,length(projection$x)) -ycenter)^2

        #output with matrix to include ellipse function
        output <- inla.mesh.project(inla.mesh.projector(mesh,
                                                        xlim=c(xini,xfin),ylim=c(yini,yfin),
                                                        dim=zoom*c(xsize+1,ysize+1)),
                                    res$summary.random$i$mean)+
            t(matrix(as.numeric(res$summary.fixed$mean[1]+
                            res$summary.fixed$mean[2]*projected_radius+
                            res$summary.fixed$mean[3]*projected_radius_2),
                nrow=zoom*(ysize+1),ncol=zoom*(xsize+1)))
        
        #output std with simple (no function)
        outputsd <- inla.mesh.project(inla.mesh.projector(mesh,xlim=c(xini,xfin),
                                                        ylim=c(yini,yfin),
                                                        dim=zoom*c(xsize+1,ysize+1)),
                                    res$summary.random$i$sd)
    
    #ellipse fct
    }
    else if(shape=='ellipse') {
    m_weights <- rep(weight, nrow(cbind(x, y)))
    covar <- cov.wt(cbind(x,y), w=m_weights)
    
        eigens <- eigen(covar$cov)
        ellipse <- (cbind(x-xcenter,y-ycenter)%*%(eigens$vectors[,1]))^2/eigens$values[1] +
            (cbind(x-xcenter,y-ycenter)%*%(eigens$vectors[,2]))^2/eigens$values[2]
        ellipse_2 =  ellipse^2

        #use parametric function of ellipse & ellipse^2 
        stk_ell <- inla.stack(data=list(par=par), A=list(A,1,1,1),
                            effects=list(i=1:spde$n.spde,
                                        m=rep(1,length(x)),ellipse=ellipse,
                                        ellipse_2=ellipse_2),tag='est')
        #caculate result model
        res <- inla(par ~ 0 + m +ellipse +ellipse_2 +f(i, model=spde),
                    data=inla.stack.data(stk_ell),
                    control.predictor=list(A=inla.stack.A(stk_ell)),scale=epar,
                    control.compute = list(openmp.strategy='huge'),
            control.inla = list(tolerance=tolerance,restart=restart))

        #print restuls
        #print(res_rad$summary.fix)

        #prjection for ellipse
        px = rep(projection$x,each=length(projection$y))
        py = rep(projection$y,length(projection$x))
        projected_ellipse <- (cbind(px-xcenter,py-ycenter)%*%(eigens$vectors[,1]))^2/eigens$values[1] +
            (cbind(px-xcenter,py-ycenter)%*%(eigens$vectors[,2]))^2/eigens$values[2]
        projected_ellipse_2 <- projected_ellipse^2

        #output with matrix to include ellipse function
        output <- inla.mesh.project(inla.mesh.projector(mesh,
                                                        xlim=c(xini,xfin),ylim=c(yini,yfin), #ch
                                                        dim=zoom*c(xsize+1,ysize+1)),#ch
                                    res$summary.random$i$mean)+
            t(matrix(as.numeric(res$summary.fixed$mean[1]+
                                res$summary.fixed$mean[2]*projected_ellipse+
                                res$summary.fixed$mean[3]*projected_ellipse_2),
                    nrow=zoom*(ysize+1),ncol=zoom*(xsize+1)))#chan

        #output std with simple (no function)
        outputsd <- inla.mesh.project(inla.mesh.projector(mesh,xlim=c(xini,xfin),
                                                        ylim=c(yini,yfin),
                                                        dim=zoom*c(xsize+1,ysize+1)),
                                    res$summary.random$i$sd)
    }
    ##NO FCT
    else if(shape=='none') {  


        stk <- inla.stack(data=list(par=par), A=list(A,1),effects=list(i=1:spde$n.spde,m=rep(1,length(x))),tag='est')

        #result
        res <- inla(par ~ 0 + m  +f(i, model=spde),
        data=inla.stack.data(stk), control.predictor=list(A=inla.stack.A(stk)),
        scale=epar)
        #print restuls
        #res_rad$summary.fix

        #output
        output <- inla.mesh.project(inla.mesh.projector(mesh,
        xlim=c(xini,xfin),ylim=c(yini,yfin),dim=zoom*c(xsize+1,ysize+1)),res$summary.random$i$mean)+
            t(matrix(as.numeric(res$summary.fixed$mean[1]),nrow=zoom*(ysize+1),ncol=zoom*(xsize+1)))
        
        #output std with simple (no function)
        outputsd <- inla.mesh.project(inla.mesh.projector(mesh,xlim=c(xini,xfin),ylim=c(yini,yfin),dim=c(xsize+1,ysize+1)),res$summary.random$i$sd)
    }    
        
   # zoom    
   # if (zoom != 1){
   #     output <- zoom_fix(output,zoom)
   #     outputsd <- zoom_fix(outputsd,zoom)
   # }
        
    #original data to compare
    xbin <- (xfin-xini)/(xsize+1)
    ybin <- (yfin-yini)/(ysize+1) 
    xmat <- (x-xini)/xbin  #map of coordinates into indices
    ymat <- (y-yini)/ybin  #map of coordinates into indices
    timage <- matrix(NA,nrow=xsize+1,ncol=ysize+1)
        for (i in 1:length(x)) {timage[xmat[i],ymat[i]] <- par[i]}
    terrimage = NULL
    if (hasArg(tepar)) {
        terrimage <- matrix(NA,nrow=xsize+1,ncol=ysize+1)
        for (i in 1:length(x)) {terrimage[xmat[i],ymat[i]] <- epar[i]}
    }

    #more info
    mim <- melt(output)
    colnames(mim) <- c("x","y","value")
    xx <- mim$x/zoom-1
    yy <- mim$y/zoom-1
    zz <- mim$value   
    sdmim <- melt(outputsd)
    colnames(sdmim) <- c("x","y","value")
    erzz <- sdmim$value

        
    return(list(out=output, image=timage, erimage=terrimage,outsd=outputsd, x=xx,y=yy,z=zz,erz=erzz))
}

plot_and_save_images <- function(pr_data, imginla, outfile, eroutfile) {
  x <- pr_data$x
  y <- pr_data$y
  logimg <- pr_data$logimg
  valid <- pr_data$valid
  ysize <- pr_data$ysize
  xsize <- pr_data$xsize
  inla_out <-imginla$out
  imginla_out <- imginla$out
  imginla_outsd <- imginla$outsd
  imginla_out[is.nan(imginla_out)] <- 0
  imginla_out[is.na(imginla_out)] <- 0
  imginla_outsd[is.nan(imginla_outsd)] <- 0
  imginla_outsd[is.na(imginla_outsd)] <- 0
  logimg[is.nan(logimg)] <- 0
  logimg[is.na(logimg)] <- 0
  cutColor <- 100
  name <- "test"
  colpal <- viridis::viridis(100)
  ## PLOT
  fj5_img <- classIntervals(c(logimg[valid], imginla_out[valid]), 
                            n = cutColor, style = "fisher")
  
  inimg <- levelplot(imginla$image, xlim = c(0, ysize), ylim = c(0, xsize),
                     col.regions = colpal,
                     main = list(name, side = 1, line = 0.5, cex = 1.4), 
                     ylab = list('y [pixels]', cex = 1.5),
                     cut = cutColor, at = fj5_img$brks,
                     xlab = list('x [pixels]', cex = 1.3))
  
  outimg <- levelplot(imginla$out, xlim = c(0, ysize), ylim = c(0, xsize),
                      col.regions = colpal, xlab = 'x', ylab = 'y',
                      cut = cutColor, at = fj5_img$brks)
  
  fj5_erimg <- classIntervals(imginla_outsd[valid], n = 100, style = "fisher")
  
  outsdimg <- levelplot(imginla$outsd, xlim = c(0, ysize), ylim = c(0, xsize),
                        col.regions = colpal, xlab = 'x', ylab = 'y',
                        cut = cutColor, at = fj5_erimg$brks)
  
  save(x, y, logimg, imginla, file = paste(outfile, 'Rdata', sep = '.'))
  writeFITSim(imginla$out, file = paste(outfile, 'fits', sep = '.'))
  writeFITSim(imginla$outsd, file = paste(eroutfile, 'fits', sep = '.'))
  
  png(filename = paste(outfile, 'png', sep = '.'))
  print(c(inimg, outimg, outsdimg, layout = c(3, 1)))
}

plot_inla <- function(inla_result, title_prefix = "INLA Result", output_dir = "plots") {
  # Create the output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  
  # Extract the components from the result
  reconstructed_image <- inla_result$out
  original_image <- inla_result$image
  error_image <- inla_result$erimage
  reconstructed_sd <- inla_result$outsd
  x <- inla_result$x
  y <- inla_result$y
  z <- inla_result$z
  erz <- inla_result$erz
  
  # Define file names
  original_image_file <- file.path(output_dir, paste0(title_prefix, "_Original_Image.png"))
  reconstructed_image_file <- file.path(output_dir, paste0(title_prefix, "_Reconstructed_Image.png"))
  error_image_file <- file.path(output_dir, paste0(title_prefix, "_Error_Image.png"))
  reconstructed_sd_file <- file.path(output_dir, paste0(title_prefix, "_Reconstruction_SD.png"))
  scatter_plot_file <- file.path(output_dir, paste0(title_prefix, "_Scatter_Plot.png"))
  error_scatter_plot_file <- file.path(output_dir, paste0(title_prefix, "_Error_Scatter_Plot.png"))
  
  # Save the original image
  png(original_image_file)
  image(original_image, col = terrain.colors(256), main = paste(title_prefix, " - Original Image"))
  dev.off()
  
  # Save the reconstructed image
  png(reconstructed_image_file)
  image(reconstructed_image, col = terrain.colors(256), main = paste(title_prefix, " - Reconstructed Image"))
  dev.off()
  
  # Save the error image if it exists
  if (!is.null(error_image)) {
    png(error_image_file)
    image(error_image, col = terrain.colors(256), main = paste(title_prefix, " - Error Image"))
    dev.off()
  }
  
  # Save the standard deviation of the reconstruction
  png(reconstructed_sd_file)
  image(reconstructed_sd, col = terrain.colors(256), main = paste(title_prefix, " - Reconstruction SD"))
  dev.off()
  
  # Save the scatter plot of x, y, z values
  png(scatter_plot_file)
  plot(x, y, col = terrain.colors(256)[cut(z, 256)], pch = 19, main = paste(title_prefix, " - Scatter Plot"))
  dev.off()
  
  # Save the scatter plot of x, y, erz values
  png(error_scatter_plot_file)
  plot(x, y, col = terrain.colors(256)[cut(erz, 256)], pch = 19, main = paste(title_prefix, " - Error Scatter Plot"))
  dev.off()
  
  # Optionally, you can also display the plots in the R console
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  image(original_image, col = terrain.colors(256), main = paste(title_prefix, " - Original Image"))
  image(reconstructed_image, col = terrain.colors(256), main = paste(title_prefix, " - Reconstructed Image"))
  if (!is.null(error_image)) {
    image(error_image, col = terrain.colors(256), main = paste(title_prefix, " - Error Image"))
  }
  image(reconstructed_sd, col = terrain.colors(256), main = paste(title_prefix, " - Reconstruction SD"))
  plot(x, y, col = terrain.colors(256)[cut(z, 256)], pch = 19, main = paste(title_prefix, " - Scatter Plot"))
  plot(x, y, col = terrain.colors(256)[cut(erz, 256)], pch = 19, main = paste(title_prefix, " - Error Scatter Plot"))
}

# Example usage:
# Assuming `result` is the output from the `stationary_inla` function
# result <- stationary_inla(tx, ty, tpar, tepar, ...)
# plot_and_save_inla_results(result)