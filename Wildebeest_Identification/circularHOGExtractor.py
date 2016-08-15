#----------------------------------------------------------------------------------------------------------------------#
# Title:    Analysis of Wildebeest aerial images
# Author:   Colin Torney + John Joseph Valletta
# Date:     18/02/2016   
# Data:     Serengeti National Park, Tanzania
# Note:     I'm using this slow version of t-SNE because there's a bug in sklearn.manifold.TSNE
# See:      https://github.com/scikit-learn/scikit-learn/issues/6204 or https://github.com/jjvalletta/t-SNEIssue/tree/master
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Preamble
#----------------------------------------------------------------------------------------------------------------------#
import cmath as cm
import numpy as np


#----------------------------------------------------------------------------------------------------------------------#
# Class definition 
#----------------------------------------------------------------------------------------------------------------------#
class circularHOGExtractor():
    """
    Create a 1D edge length histogram and 1D edge angle histogram.
    
    This method takes in an image, applies an edge detector, and calculates
    the length and direction of lines in the image.
    
    bins = the number of bins
    """
    def __init__(self, bins=4, size=6, max_freq=4):

        # number of bins in the radial direction for large scale features
        self.mNBins = bins
        # size of bin in pixels, this sets the required radius for the image = bins*size
        self.mNSize = size
        # number of fourier modes that will be used (0:modes-1)
        self.mNMaxFreq = max_freq 

        mf = self.mNMaxFreq+1
        self.mNCount = 2*(bins-1) * (mf + 2*(np.dot([mf - i for i in range(mf)] , range(mf))  ))
        # create a list to store kernels for regional descriptors based on circular harmonics
        self.ciKernel = []

        # first create the central region 
        [x,y]=np.meshgrid(range(-self.mNSize+1,self.mNSize),range(-self.mNSize+1,self.mNSize))
        z = x + 1j*y
        kernel = self.mNSize - np.abs(z)
        kernel[kernel < 0] = 0
        kernel = kernel/sum(sum(kernel))

 #       self.ciKernel.append(kernel)

        # next build the internal regions - (bins-1) concentric circles
        modes = range(0, self.mNMaxFreq+1)
        scale = range(2, self.mNBins+1)

        for s in scale:
            r = int(self.mNSize * s)
            ll = range(1-r,r)
            [x,y] = np.meshgrid(ll,ll)
            z = x + 1j*y
            phase_z = np.angle(z);
                            
            for k in modes:
                kernel = self.mNSize - np.abs(np.abs(z) - (r-self.mNSize)) 
                kernel[kernel < 0] = 0
                kernel = np.multiply(kernel,np.exp(1j*phase_z*k))
                sa = np.ravel(np.abs(kernel))
                kernel = kernel / np.sqrt(np.sum(np.multiply(sa,sa)))

                self.ciKernel.append(kernel)



    def extract(self, img):
        I = img.astype(float)/255.0

	#minI = np.min(I)
	#maxI = np.max(I)
	#I = (I-minI)/(maxI-minI)
        I = (I-I.mean())/I.std()

        # size and centre of image
        (nx, ny) = I.shape
        cx = int(round(0.5*nx))
        cy = int(round(0.5*ny))

        # compute gradient with a central difference method and store in complex form
        (dy, dx) = np.gradient(I)
        dz = dx + 1j*dy

        # compute magnitude/phase of complex numbers
        phi = np.angle(dz)
        r = np.abs(dz)
        #r = r/(r.std()+0.0001)


        # create an empty array for storing the dfft of the orientation vector
        histF = np.zeros([nx, ny, self.mNMaxFreq+1])+0j

        # take the dfft of the orientation vector up to order MaxFreq
        # positive values of k only since negative values give conjugate
        for k in range(0,self.mNMaxFreq+1):
            histF[:,:,k] = np.multiply(np.exp( -1j * (k) * phi) , r+0j)
        

        # compute regional descriptors by convolutions (these descriptors are not rotation invariant)
        fHOG = np.zeros([self.mNCount])
        scale = range(0, self.mNBins-1)
        f_index = 0
        for s in scale:
            allVals = np.zeros((self.mNMaxFreq+1,self.mNMaxFreq+1),dtype=np.complex64)
            for freq in range(0,self.mNMaxFreq+1):
                template = self.ciKernel[s*(self.mNMaxFreq+1)+freq]
                (tnx, tny) = template.shape
                tnx2 = int(round(0.5*tnx))
                for k in range(0,self.mNMaxFreq+1):
                    allVals[freq,k] = np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template)))
            for (x,y), val in np.ndenumerate(allVals):
                if x==y:
                    fHOG[f_index]=val.real
                    f_index+=1
                    fHOG[f_index]=val.imag
                    f_index+=1
                else:
                    for (x1,y1), val1 in np.ndenumerate(allVals):
                        if x1<x: continue
                        if y1<y: continue
                        if (x-y)==(x1-y1):
                            fHOG[f_index]=(val*val1.conjugate()).real
                            f_index+=1
                            fHOG[f_index]=(val*val1.conjugate()).imag
                            f_index+=1

        return fHOG.tolist()


    def prepareExtract(self, img):
        I = img.astype(float)/255.0
#      I = (I-I.mean())/I.std()

        # size and centre of image
        (nx, ny) = I.shape
        # compute gradient with a central difference method and store in complex form
        (dy, dx) = np.gradient(I)
        dz = dx + 1j*dy

        # compute magnitude/phase of complex numbers
        phi = np.angle(dz)
        r = np.abs(dz)
 #       r = r/(r.mean()+0.001)


        # create an empty array for storing the dfft of the orientation vector
        histF = np.zeros([nx, ny, self.mNMaxFreq+1])+0j

        # take the dfft of the orientation vector up to order MaxFreq
        # positive values of k only since negative values give conjugate
        for k in range(0,self.mNMaxFreq+1):
            histF[:,:,k] = np.multiply(np.exp( -1j * (k) * phi) , r+0j)

        return histF
        
    def denseExtract(self, histF, positions, N):
 #       I = img.astype(float)/255.0
#      I = (I-I.mean())/I.std()

        # size and centre of image
        (nx, ny, kk) = histF.shape
        
        features = np.zeros((N,self.mNCount),dtype=np.float32)
        scale = range(0, self.mNBins-1)
        for p in range(N):
            cx = positions[p,0]+1
            cy = positions[p,1]+1
            if cx<self.mNBins*self.mNSize: continue
            if cy<self.mNBins*self.mNSize: continue
            if cx> nx - self.mNBins*self.mNSize: continue
            if cy> ny - self.mNBins*self.mNSize: continue
            
            f_index = 0
            for s in scale:
                allVals = np.zeros((self.mNMaxFreq+1,self.mNMaxFreq+1),dtype=np.complex64)
                
                for freq in range(0,self.mNMaxFreq+1):
                    template = self.ciKernel[s*(self.mNMaxFreq+1)+freq]
                    (tnx, tny) = template.shape
                    tnx2 = int(round(0.5*tnx))
                    for k in range(0,self.mNMaxFreq+1):
                        allVals[freq,k] = np.sum(np.sum(np.multiply(histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k],template)))
                        #if p==2193 and freq==0 and s==0:
                        #        print k
                        #        for kk in histF[cx-tnx2:cx-tnx2+tnx,cy-tnx2:cy-tnx2+tnx,k]:
                        #            for jj in kk:
                        #                print jj.real
                
                
                for (x,y), val in np.ndenumerate(allVals):
                    if x==y:
                        features[p,f_index]=val.real
                        f_index+=1
                        features[p,f_index]=val.imag
                        f_index+=1

                    else:
                        for (x1,y1), val1 in np.ndenumerate(allVals):
                            if x1<x: continue
                            if y1<y: continue
                            if (x-y)==(x1-y1):
                                features[p,f_index]=(val*val1.conjugate()).real
                                f_index+=1
                                features[p,f_index]=(val*val1.conjugate()).imag
                                f_index+=1

        
        return features

#        print "diff to original array:"
#        print features[0], fHOG[0]
#        print np.max(np.abs(features-fHOG))

        return fHOG.tolist()

    
    def getFieldNames(self):
        """
        Return the names of all of the length and angle fields. 
        """
        retVal = []
        for i in range(0,self.mNCount):
            name = "Length"+str(i)
            retVal.append(name)
                        
        return retVal
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """

    def getNumFields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.mNCount


