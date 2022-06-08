import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import play.api.libs.iteratee.internal;

//application permettante de creer le model
public class CNNModelMnist {

	public static void main(String[] args) throws IOException, InterruptedException {//#1
		
		long seed=1234;//definition du seed//#3
		double learningRate=0.001;//#4
		long height=28;//hauteur de l'image est de 28 pixel//#5
		long weght=28;//largeur de l'image//#6
		long depth=1;//la profondeur de l'image est de 1 pour une image noir sur blanc et 3 pour une image RGB//#7
		int outputsize=10; //on dfinir le nbre de noeud en sortie representant les image de 1 a 10
		int batchsize=54; //pour entrainner le model, on prend 54 images en entree et on donne 54 sortie
		int  classIndex=1; //represente le nbre d'entree du model
		
		//**************creation du modele********************
		//*************2
		System.out.println("***********Creation du model**********");
		//creation de la configuration du model
		MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
				.seed(seed) //creation du seed pour ne pas fait varier les valeurs aleatoires apres chaque execution du programme
				.updater(new Adam(learningRate)) //on definir Adam comme algorithme de retro-propagation
				.list()
				.setInputType(InputType.convolutionalFlat(height, weght, depth))//la couche convolutionFlat parceque se sont des images contenu dans le dataSet
				.layer(0,new ConvolutionLayer.Builder() //creationde la couche d convolution 0
						.nIn(depth) //on donne l'entree a la couche de convolution et comme c'est une image on donne la profondeur de l'image
						.nOut(20) //on donne la sortie, pour chaque image on utilisera 20 filtre(Kernel) et donc on obtiendra 20image filtree
						.activation(Activation.RELU)//fonction d'action
						.kernelSize(5,5) //taille du filtre de convolution
						.stride(1,1) //on donne le stride pour le deplacement du filtre
						.build())
				
				
				.layer(1,new SubsamplingLayer.Builder()//creation de la couche MaxPolling 1
						.kernelSize(2,2)//filtre du  polling
						.stride(2,2)//deplacement du filtre
						.poolingType(SubsamplingLayer.PoolingType.MAX) //on definir comme type de pooling le MaxPooling mais il y'a n'a d'autre
						.build())
				
				
				
				.layer(2,new ConvolutionLayer.Builder()//creationde la couche d convolution 2
						//pas besion de preciser le nbre d'entree car on utilise un couche fullyConnected
						.nOut(50)//on donne la sortie, pour chaque image on utilisera 50 filtre(Kernel) et donc on obtiendra 50 image filtree
						.activation(Activation.RELU)
						.kernelSize(5,5)
						.stride(1,1)
						.build())
				.layer(3,new SubsamplingLayer.Builder()//creation de la couche MaxPolling 2
						
						.poolingType(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				
				
				.layer(4,new DenseLayer.Builder()//creation de  la couche fullyConnected
						.nOut(500)
						.activation(Activation.RELU)
						.build()) 
				.layer(5,new OutputLayer.Builder()//creation de l couche output
						.nOut(outputsize)//resultat en sortie
						.activation(Activation.SOFTMAX) //softmax pour classer les sortie en probabilite
						.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //fonction pour minimiser les ertes
						.build())
				.build();
		
		System.out.println(configuration.toJson());//on affiche la configuratin du model
		
		
		MultiLayerNetwork model=new MultiLayerNetwork(configuration);//on cree l model
		model.init(); //on initialise le model
		
		
		
		//*************Entrainnement du model
		
		System.out.println("Model Training....");
		//on utilise la librarie DataVec pour recuperer les donnees du dataSet
		String path=System.getProperty("user.home")+"/mnist";//on recupere le chemin vers le dossier utilisateur: /user/blanc/mnist
		File fileTrain=new File(path+"/training");//on recupere le contenu du dossier user/blanc/mnist/training qui est le dataSet pour l'entrainnement du model
		FileSplit fileSplitTrain=new FileSplit(fileTrain,NativeImageLoader.ALLOWED_FORMATS,new Random(seed));//avec cet objet on donne le fileTrain et une enumeration contenant les format des images
		RecordReader recordReaderTrain=new ImageRecordReader(height,weght,depth,new ParentPathLabelGenerator());//on charge l fichir comme fichier Image, et comm dernier parametre on precise le dossier parent de l'image
		recordReaderTrain.initialize(fileSplitTrain);//on initialise le fichier
		DataSetIterator dataSetIteratorTrain=new RecordReaderDataSetIterator(recordReaderTrain, batchsize,classIndex,outputsize); //
		DataNormalization scaler=new ImagePreProcessingScaler(0,1);//on normalise ls valeur de variation entre de 0 et 1 pour eviter les pb de surApprentissage
		dataSetIteratorTrain.setPreProcessor(scaler);
		
		
		//creation de l'interface graphique
		/*UIServer uiServer=UIServer.getInstance();
		StatsStorage statsStorage=new InMemoryStatsStorage();//on stocke les statistique dans la memoire
		uiServer.attach(statsStorage); //on attache les statistiques a l'interface graphique 
		model.setListeners(new StatsListener(statsStorage));//on met l'interface graphique a l'ecoute de la memoire pour les statistique entrantes
		*/
		//apprentissage du model
		int numEpoch=1;
		for(int i=0; i<numEpoch;i++)
		{
			model.fit(dataSetIteratorTrain);
		}
		
		//pour voir comment est structuree le dataSet
		/*while (dataSetIteratorTrain.hasNext()) {
			
			DataSet dataSet=dataSetIteratorTrain.next();
			INDArray features=dataSet.getFeatures();
			INDArray labels=dataSet.getLabels();
			System.out.println(features.shapeInfoToString());
			System.out.println(labels.shapeInfoToString());
			System.out.println("------------------------------");
			
		}*/
		//***********************Evaluation du model
		
		System.out.println("Model Evaluation");
		
		//chargement du fichier de test grace a la library DataVec
		File fileTest=new File(path+"/testing");
		FileSplit fileSplit=new FileSplit(fileTest,NativeImageLoader.ALLOWED_FORMATS,new Random(seed));
		RecordReader recordReaderTest=new ImageRecordReader(height,weght,depth,new ParentPathLabelGenerator());
		recordReaderTest.initialize(fileSplit);
		DataSetIterator dataSetIteratorTest= new RecordReaderDataSetIterator(recordReaderTest,batchsize,classIndex,outputsize);
		DataNormalization scalerTest=new ImagePreProcessingScaler(0,1);
		dataSetIteratorTest.setPreProcessor(scalerTest);
		
		Evaluation evaluation=new Evaluation();
		
		while (dataSetIteratorTest.hasNext()) {
			
			DataSet dataSet=dataSetIteratorTest.next();//on recupere une ligne dans le dataSet
			INDArray features= dataSet.getFeatures();//on recupere les entree dans la ligne du tensor
			INDArray targetLabels=dataSet.getLabels();//on recupere la sortie dans la ligne du tensor
			
			INDArray predicted=model.output(features);//on recupere laprediction correspondant a l'entree
			
			evaluation.eval(predicted, targetLabels);//on fail l'evaluation en donnat la sortie attendu et la sortie obtenu
		}
		
		System.out.println(evaluation.stats());//on affiche les statistique contenant les precisions t la matrices de confusion
		ModelSerializer.writeModel(model, "CNNModelMnist.zip", true); //on enregistre le modeldans le fichier IrisModel.zip apres son evaluation et on precise qu'a chaque chargement du model on utilise le meme Updater
		
		
	}
}
