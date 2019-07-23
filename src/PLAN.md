# PLAN OF ACTION

# To Do List 

  1.) #COMPLETED:# implement a capture class that will save the pixels arrays as some sort of zipped file. 
      storing all the arrays in one go may be too big for a pickle file to store. so it needs a batch size or try saving one at a time
      
      add data augmenetation features so that the blocks dont look completely the same so set some randomly to jitter the width and
      the hieghts of the top and bottom blocks but only by a little too much jitter may be unstable
      
      ADDED: Rescale pygame function now the a copy of the screen can be downsampled to very samll resolution without effecting
      any of the game mechanics, so the simulation can run on 800x600 screen but the VAE will get a heavly downsampled version to spped up
      processing time. everything scales with this function aswell which is AMAZING!!!!!  
      lowest setting are 40x40 any lower than this and some of the blockobjects width start warping. 
      
 2.)
 
     Use capture images on VAE and see how it responds 
