import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

class Instance{
    double[] Attributes;
    String Label;
    Instance(double[] attrs, String l){
        this.Attributes = attrs;
        this.Label = l;

    }
    Instance(){
        this.Attributes = new double[classify.instance_num];
        this.Label = ""; // means only been initialized
    }
    void print_info(){
        for(double d : Attributes){
            System.out.print((float)d+", ");
        }
        System.out.println(Label);
    }
}

public class classify {
    static double STEP_SIZE = 0.001;    // step size for gradient descent
    static double EPSILON = 0.0001;      // epsilon for gradient descent
    static double M = 1.0;    // M for filtering out the point which is so far away from Instance
    static int N = 100;   // num of tries remain
    static String output_format_flag = "";
    static ArrayList<Instance> Training_Set = new ArrayList<>();

    static HashSet<String> Label_Set = new HashSet<>();
    static int instance_num;
    static int ofs = 0;
    static int rnum = 0;

    public static void main(String[] args) throws IOException {
        //System.out.println(args.length);
        if(!(args.length == 5 || args.length == 6)){
             System.exit(0);
        }

        get_input_from(args[0]);
        STEP_SIZE = Double.parseDouble(args[1]);
        EPSILON =  Double.parseDouble(args[2]);
        M =  Double.parseDouble(args[3]);
        N = Integer.parseInt(args[4]);
        if(args.length == 6){
            output_format_flag = args[5];
        }
        //for(Instance i : Training_Set){
        //    i.print_info();
        //}
        //System.out.println();
        // get the label set from training set
        for(Instance i : Training_Set){
            Label_Set.add(i.Label);
        }

        double max_accuracy =  0;
        ArrayList<Instance> final_trained_Examplar = new ArrayList<>();
        for(int i=0; i<N+1; i++){
            ArrayList<Instance> Trained_Examplar = gradDescent_driver(Training_Set, STEP_SIZE, EPSILON, M, i);
            double Final_Accuracy = compute_final_accuracy(Training_Set, Trained_Examplar);
            if(output_format_flag.equals("-v")) {
                System.out.print("Converged. Accuracy = "+(float)Final_Accuracy+"\n");
                System.out.println();
            }
            if(Final_Accuracy > max_accuracy) {
                max_accuracy = Final_Accuracy;
                final_trained_Examplar = Trained_Examplar;
            }
        }
        System.out.print("Best Accuracy = " + (float)max_accuracy + "\n");
        System.out.print("Final Trained Examplars:\n");
        for(Instance e : final_trained_Examplar){
            e.print_info();
        }
    }

    static void get_input_from(String filename) throws IOException {
        File file  = new File(filename);
        BufferedReader bf = new BufferedReader(new FileReader(file));
        String line;
        while((line = bf.readLine())!=null){
            String temp[] =  line.split(",");
            instance_num = temp.length - 1;
            double attrs[] = new double[instance_num];
            for(int i =0; i<instance_num; i++) {
                attrs[i] = Double.parseDouble(temp[i]);
            }
            Instance inst = new Instance(attrs, temp[temp.length-1]);
            Training_Set.add(inst);
        }
    }

    static ArrayList<Instance> gradDescent_driver(ArrayList<Instance> Training_Set, double step_size, double epsilon, double m,int num_tries_remain){
        ArrayList<Instance> Exemplar_List = new ArrayList<>();
        if(num_tries_remain == 0){  // first try start Instances at centroids
            System.out.print("\tRound 0: Centroids as Exemplars\n");
            for(String l : Label_Set){
                double centroid[] = new double[instance_num];
                double num_inst = 0; // number of instance have label l
                for(Instance i : Training_Set){
                    if(i.Label.equals( l)){
                        num_inst += 1;
                        for(int j=0; j<instance_num; j++){
                            centroid[j] += i.Attributes[j];
                        }
                    }
                }
                for(int j=0; j<instance_num; j++){
                    centroid[j] = centroid[j]/num_inst;
                }
                Instance e = new Instance(centroid, l);
                Exemplar_List.add(e);
            }
            rnum+=1;
            //System.out.print("Converged. Accuracy = "+Final_Accuracy);
        }else{  // random start
            ofs = 0;
            System.out.print("\tRound " + rnum +": Random restart\n");
            for(String l : Label_Set) {
                double up_boundry[] = new double[instance_num];
                double low_boundry[] = new double[instance_num];
                for (int i = 0; i < instance_num; i++) {// initialize upper boundry and lower boundry
                    up_boundry[i] = Double.NEGATIVE_INFINITY;
                    low_boundry[i] = Double.POSITIVE_INFINITY;
                }
                for (Instance t : Training_Set) {
                    if(t.Label.equals(l)) {
                        for (int i = 0; i < instance_num; i++) {
                            if (t.Attributes[i] > up_boundry[i]) {
                                up_boundry[i] = t.Attributes[i];
                            }
                            if (t.Attributes[i] < low_boundry[i]) {
                                low_boundry[i] = t.Attributes[i];
                            }
                        }
                    }
                }
                Random rand = new Random();
                double rand_exemplar_attr[] = new double[instance_num];
                for(int i=0; i<instance_num; i++){
                    rand_exemplar_attr[i] = low_boundry[i]+ rand.nextDouble()*(up_boundry[i] - low_boundry[i]);
                }
                Instance rand_exemplar = new Instance(rand_exemplar_attr, l);
                Exemplar_List.add(rand_exemplar);
            }
            //System.out.print("Converged. Accuracy = "+Final_Accuracy);
            rnum+=1;
        }
        return gradDescent(Training_Set, step_size, epsilon, m, Exemplar_List);
    }

    static ArrayList<Instance> gradDescent(ArrayList<Instance> Training_Set, double step_size, double epsilon, double m, ArrayList<Instance> Exemplar_List){
        double prevCost = Double.POSITIVE_INFINITY;
        double prevAccuracy = compute_accuracy(Training_Set, Exemplar_List);
        //System.out.println(prevAccuracy);
        while(true){
            double TotalCost = 0.0;
            ArrayList<Instance> neg_gradient_list = new ArrayList<>(); // negative gradient list
            for(Instance e : Exemplar_List){
                Instance temp = new Instance(new double[instance_num], e.Label);
                neg_gradient_list.add(temp);
            }

            for(Instance y : Training_Set){
                // get the real Instance
                Instance exem_v = new Instance();
                for(Instance e : Exemplar_List){
                    if(e.Label.equals(y.Label)){
                        exem_v.Label = e.Label;
                        exem_v.Attributes = e.Attributes.clone();
                    }
                }
                // find Instance closest to y.attributes
                Instance exem_w = new Instance();
                double min_dist =  Double.POSITIVE_INFINITY;
                for(Instance e : Exemplar_List){
                    double dist = compute_distance(e.Attributes, y.Attributes);
                    if(min_dist > dist){
                        exem_w.Label = e.Label;
                        exem_w.Attributes = e.Attributes.clone();
                        min_dist = dist;
                    }
                }
                if(exem_w.Label.equals(y.Label) == false){
                    double cost = compute_distance(exem_v.Attributes, y.Attributes) - compute_distance(y.Attributes,exem_w.Attributes);
                    if(cost < M){
                        for(Instance n : neg_gradient_list){
                            if(n.Label.equals(exem_v.Label)){
                                for(int i=0; i<instance_num; i++){
                                    n.Attributes[i] += y.Attributes[i] - exem_v.Attributes[i];
                                }
                            }
                            if(n.Label.equals(exem_w.Label)){
                                for(int i=0; i<instance_num; i++){
                                    n.Attributes[i] += exem_w.Attributes[i] - y.Attributes[i];
                                }
                            }
                        }
                        TotalCost += cost;
                    }else{
                        TotalCost += M;
                    }
                }
            }
            if (TotalCost < epsilon) {
                //return prevAccuracy;
                return Exemplar_List;
            }
            if(TotalCost >(1-epsilon)*prevCost){
                //return  prevAccuracy;
                return Exemplar_List;
            }
            ArrayList<Instance> prevExemplar_List = new ArrayList<>();
            for(Instance e : Exemplar_List){
                double[] v = new double[instance_num];
                for(int i=0; i< instance_num; i++){
                    v[i] = e.Attributes[i];
                }
                Instance temp = new Instance(v, e.Label);
                prevExemplar_List.add(temp);
            }
            for(int i=0; i<Exemplar_List.size();i++){
                for(int j=0; j<instance_num;j++) {
                    Exemplar_List.get(i).Attributes[j] += step_size * neg_gradient_list.get(i).Attributes[j];
                }
            }
            double newAccuracy = compute_accuracy(Training_Set, Exemplar_List);
            if(newAccuracy < prevAccuracy){
                //return prevAccuracy;
                return prevExemplar_List;
            }
            prevCost = TotalCost;
            prevAccuracy = newAccuracy;
        }
    }

    static double compute_distance(double[] v1, double[] v2){   // actually compute the squared distance
        double dist = 0.0;
        for(int i=0; i<v1.length; i++){
            dist += (v1[i]-v2[i])*(v1[i]-v2[i]);
        }
        return dist;
    }

    static double compute_accuracy(ArrayList<Instance> Training_Set, ArrayList<Instance> Exemplar_List){
        double true_positive = 0.0;
        for(Instance y : Training_Set){
            double min_dist = Double.POSITIVE_INFINITY;
            String predict_label = "";
            for(Instance e : Exemplar_List){
                double dist = compute_distance(y.Attributes, e.Attributes);
                if(min_dist > dist){
                    predict_label = e.Label;
                    min_dist = dist;
                }
            }
            if(predict_label.equals(y.Label)){
                true_positive += 1;
            }
        }
        if(output_format_flag.equals("-v")) {
            System.out.print(" Iteration: " + ofs + "\n");
            for (Instance e : Exemplar_List) {
                e.print_info();
            }
            System.out.println("Accuracy " + (float) true_positive / Training_Set.size());
            System.out.println();
        }
        ofs += 1;
        return true_positive / Training_Set.size();
    }

    static double compute_final_accuracy(ArrayList<Instance> Training_Set, ArrayList<Instance> Exemplar_List){
        double true_positive = 0.0;
        if (!output_format_flag.equals("-v")){
            System.out.print("Correctly clasified instances:\n");
        }
        for(Instance y : Training_Set) {
            double min_dist = Double.POSITIVE_INFINITY;
            String predict_label = "";
            for (Instance e : Exemplar_List) {
                double dist = compute_distance(y.Attributes, e.Attributes);
                if (min_dist > dist) {
                    predict_label = e.Label;
                    min_dist = dist;
                }
            }
            if (predict_label.equals(y.Label)) {
                if(!output_format_flag.equals("-v")){
                    y.print_info();
                }
                true_positive += 1;
            }
        }
        if (!output_format_flag.equals("-v")){
            System.out.println();
        }
        return true_positive / Training_Set.size();
    }
}
