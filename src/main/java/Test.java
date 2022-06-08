import java.util.Random;
//classe permettante de generer les valeurs aleatoires et et voir cmt fonctionne le seed
public class Test {

	public static void main(String[] args) {
		
		Random random=new Random(1234);//avec la precision du seed les nombres aleatoire ne changerons pas a chaque execution
		
		//chacune des instructons renverra un nbre aleatoire entre 0 et 20 êt ce nbre changear a chaque execution lorsque la valeur du seed ne sera preciser
		System.out.println(random.nextInt(20));
		System.out.println(random.nextInt(20));
		System.out.println(random.nextInt(20));


	}
}
