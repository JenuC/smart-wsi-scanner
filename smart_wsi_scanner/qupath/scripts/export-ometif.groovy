// source https://forum.image.sc/t/issue-with-bfconvert-and-big-jpg-images/43276/2
// and https://forum.image.sc/t/using-qupath-to-convert-very-large-tif-to-ome-tif-and-add-metadata/48636/4
// With grand struggles from Mike Nelson
// QuPath 0.2.3+

server = getCurrentServer()
//take arguments from the command line for pixel size and magnification. Other arguements could be added to the list.
pixelSize = Double.parseDouble(args[0])
magnification = Double.parseDouble(args[1])
// I found that server.getFile() fails for this type of image, and I needed to use
// For other file types, this might not all be necessary.
path= server.getBuilder().getURIs()[0].toString()
imageData = getCurrentImageData()
baseFilePath = path.substring(6, path.lastIndexOf(".")+1)
def pathInput = baseFilePath + "tif"
def pathOutput = baseFilePath + "ome.tif"

println 'Reading image...'
//def img = ij.IJ.openImage(pathInput).getBufferedImage()
//def server = new WrappedBufferedImageServer("Anything", img)
def metadataNew = new ImageServerMetadata.Builder(server.getMetadata())
    .pixelSizeMicrons(pixelSize, pixelSize)
    .zSpacingMicrons(1)
	.magnification(magnification)
    .build();
    
imageData.updateServerMetadata(metadataNew);
println 'Writing OME-TIFF'
new OMEPyramidWriter.Builder(server)
        .parallelize()
        .tileSize(512)
        .scaledDownsampling(1, 4)
        .build()
        .writePyramid(pathOutput)
        
println 'Done!'


import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.images.servers.*
import javax.imageio.*
import qupath.lib.images.servers.ImageServerMetadata;