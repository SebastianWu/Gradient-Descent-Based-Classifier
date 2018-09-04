# Gradient Descent Based Classifier
Implement a gradient-based classifier for a classification problem with numerical predictive attributes and a discrete classification attribute.

Author: Yuanxu Wu  

# How to run this program
## Prerequisites  
1. Java 1.8.0  

## How to compile and run the three programs:  
1. $javac classify.java  
2. $java classify <input file name> <step size> <epsilon> <M> <N> <-optional output format flag>  
for example: $java classify test1.txt 0.1 0.1 2 0 -v   
or $java classify test1.txt 0.1 0.1 2 0  

## Input txt file format:  
1,1,2,a  
2,1,1,a  
2,0,1,a  
0,2,1,b  
3,2,0,b  
3,3,0,c  
0,3,0,c  
3,2,1,c  
0,3,3,c  

## Example:  
$java classify test1.txt 0.1 0.1 2 0  
	Round 0: Centroids as Exemplars  
Correctly clasified instances:  
1.0, 1.0, 2.0, a  
2.0, 1.0, 1.0, a  
2.0, 0.0, 1.0, a  
0.0, 2.0, 1.0, b  
3.0, 2.0, 0.0, b  
3.0, 3.0, 0.0, c  
3.0, 2.0, 1.0, c  
0.0, 3.0, 3.0, c  

Best Accuracy = 0.8888889  
Final Trained Examplars:  
1.6666666, 0.6666667, 1.3333334, a  
1.35, 2.0, 0.45, b  
1.65, 2.675, 1.0, c  

  
