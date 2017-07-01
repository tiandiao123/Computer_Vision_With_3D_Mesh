* Randomly sample images' pixels and texture mapping 
* try to get texture working well
* move camera, save views and cameras' pose 

###
（1） fix texturing: normal tissue, create a texture as realistic as possible, compare the original image
(2) grab camera pose, trying to unsderstand where we put the camera
(3) replace light source with a light source on top of camera

        List<String> list=new ArrayList<>();
        int w=0;
        for(int i=0;i<words.length;i++){
            int len=-1;
            for(w=i;w<words.length&&len+words[w].length()+1<=maxWidth;w++){
                len+=words[w].length()+1;
            }
            
            int space=(w!=i+1) ? (maxWidth/(w-i-1)+1) : 1;
            int extra=(w!=i+1) ? maxWidth%(w-i-1) : 0;
            StringBuilder sb=new StringBuilder();
            sb.append(words[i]);
            for(int j=i+1;j<w;j++){
                for(int k=0;k<space;k++){
                    sb.append(" ");
                }
                if(extra-->0){
                    sb.append(" ");
                }
                sb.append(words[j]);
            }
            
            int strlen=maxWidth-sb.toString().length();               
            while(strlen--!=0){
                sb.append(" ");
            }
            list.add(sb.toString());
            i=w-1;
        }
        
        return list;