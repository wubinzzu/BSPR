package librec.util;


import static librec.util.Gamma.digamma;
import librec.data.AddConfiguration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.util.FileIO;
import librec.util.Logs;
import librec.util.Randoms;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

/**
 * Latent Dirichlet Allocation for implicit feedback: Tom Griffiths, <strong>Gibbs sampling in the generative model of
 * Latent Dirichlet Allocation</strong>, 2002. <br>
 * 
 * <p>
 * <strong>Remarks:</strong> This implementation of LDA is for implicit feedback, where users are regarded as documents
 * and items as words. To directly apply LDA to explicit ratings, Ian Porteous et al. (AAAI 2008, Section Bi-LDA)
 * mentioned that, one way is to treat items as documents and ratings as words. We did not provide such an LDA
 * implementation for explicit ratings. Instead, we provide recommender {@code URP} as an alternative LDA model for
 * explicit ratings.
 * </p>
 * 
 * @author Guibing Guo
 *
 */
@AddConfiguration(before = "factors, alpha, beta")
public class LDA {
	private SparseMatrix corpus;
	protected int numDocs, numWords;
	
	private DenseMatrix PukSum, PkiSum;
	private DenseMatrix Puk, Pki;
	private DenseMatrix Nuk, Nki;
	private DenseVector Nu, Nk;
	private DenseVector alpha, beta;
	
	protected Table<Integer, Integer, Integer> z;
	
	protected static int numFactors, numIters, burnIn, sampleLag, numIntervals;
	protected static double initAlpha, initBeta;
	protected static String outPath;
	/**
	 * size of statistics
	 */
	protected int numStats = 0;
	protected String foldInfo;
	public LDA(SparseMatrix _corpus) {
		corpus = _corpus;
	}
	
	public void setParameters(int _numFactors, int _numIters, int _burnIn, int _sampleLag, int _numIntervals, 
			double _initAlpha, double _initBeta, String _outPath, String _foldInfo) throws Exception {
		numDocs = corpus.numRows();
		numWords = corpus.numColumns();
		numFactors = _numFactors;
		numIters = _numIters;
		burnIn = _burnIn;
		sampleLag = _sampleLag;
		numIntervals = _numIntervals;
		initAlpha = (float)_initAlpha;
		initBeta = (float)_initBeta;
		outPath = _outPath;
		foldInfo = _foldInfo;
	}

	protected void initModel() throws Exception {

		PukSum = new DenseMatrix(numDocs, numFactors);
		PkiSum = new DenseMatrix(numFactors, numWords);

		// initialize count variables.
		Nuk = new DenseMatrix(numDocs, numFactors);
		Nu = new DenseVector(numDocs);

		Nki = new DenseMatrix(numFactors, numWords);
		Nk = new DenseVector(numFactors);

		alpha = new DenseVector(numFactors);
		alpha.setAll(initAlpha);

		beta = new DenseVector(numWords);
		beta.setAll(initBeta);

		// The z_u,i are initialized to values in [0, K-1] to determine the initial state of the Markov chain.
		z = HashBasedTable.create();
		for (MatrixEntry me : corpus) {
			int d = me.row();
			int w = me.column();
			int t = (int) (Randoms.uniform() * numFactors); // 0 ~ k-1

			// assign a topic t to pair (u, i)
			z.put(d, w, t);

			// number of items of user u assigned to topic t.
			Nuk.add(d, t, 1);
			// total number of items of user u
			Nu.add(d, 1);
			// number of instances of item i assigned to topic t
			Nki.add(t, w, 1);
			// total number of words assigned to topic t.
			Nk.add(t, 1);
		}
	}
	
	public void buildModel() throws Exception {
		Logs.debug("LDA{} Intializing ...\n", foldInfo);
		initModel();
		
		
		Logs.debug("LDA{} Training ...\n", foldInfo);
		for (int iter = 1; iter <= numIters; iter++) {

			// E-step: infer parameters
			eStep();

			// M-step: update hyper-parameters
			mStep();

			// get statistics after burn-in
			if ((iter > burnIn) && (iter % sampleLag == 0)) {
				readoutParams();

			}

			if (iter % numIntervals == 0)
				Logs.debug("{}{} runs at iter {}/{}", "LDA", foldInfo, iter, numIters);
		}

		// retrieve posterior probability distributions
		estimateParams();
	}

	protected void eStep() {

		double sumAlpha = alpha.sum();
		double sumBeta = beta.sum();

		// Gibbs sampling from full conditional distribution
		for (MatrixEntry me : corpus) {
			int d = me.row();
			int w = me.column();
			int t = z.get(d, w); // topic

			Nuk.add(d, t, -1);
			Nu.add(d, -1);
			Nki.add(t, w, -1);
			Nk.add(t, -1);

			// do multinomial sampling via cumulative method:
			double[] p = new double[numFactors];
			for (int k = 0; k < numFactors; k++) {
				p[k] = (Nuk.get(d, k) + alpha.get(k)) / (Nu.get(d) + sumAlpha) * (Nki.get(k, w) + beta.get(w))
						/ (Nk.get(k) + sumBeta);
			}
			// cumulating multinomial parameters
			for (int k = 1; k < p.length; k++) {
				p[k] += p[k - 1];
			}
			// scaled sample because of unnormalized p[], randomly sampled a new topic t
			double rand = Randoms.uniform() * p[numFactors - 1];
			for (t = 0; t < p.length; t++) {
				if (rand < p[t])
					break;
			}

			// add newly estimated z_i to count variables
			Nuk.add(d, t, 1);
			Nu.add(d, 1);
			Nki.add(t, w, 1);
			Nk.add(t, 1);

			z.put(d, w, t);
		}

	}

	protected void mStep() {
		double sumAlpha = alpha.sum();
		double sumBeta = beta.sum();
		double ak, bi;

		// update alpha vector
		for (int k = 0; k < numFactors; k++) {

			ak = alpha.get(k);
			double numerator = 0, denominator = 0;
			for (int u = 0; u < numDocs; u++) {
				numerator += digamma(Nuk.get(u, k) + ak) - digamma(ak);
				denominator += digamma(Nu.get(u) + sumAlpha) - digamma(sumAlpha);
			}
			if (numerator != 0)
				alpha.set(k, ak * (numerator / denominator));
		}

		// update beta_k
		for (int i = 0; i < numWords; i++) {

			bi = beta.get(i);
			double numerator = 0, denominator = 0;
			for (int k = 0; k < numFactors; k++) {
				numerator += digamma(Nki.get(k, i) + bi) - digamma(bi);
				denominator += digamma(Nk.get(k) + sumBeta) - digamma(sumBeta);
			}
			if (numerator != 0)
				beta.set(i, bi * (numerator / denominator));
		}
	}

	/**
	 * Add to the statistics the values of theta and phi for the current state.
	 */
	protected void readoutParams() {
		double sumAlpha = alpha.sum();
		double sumBeta = beta.sum();

		double val = 0;
		for (int u = 0; u < numDocs; u++) {
			for (int k = 0; k < numFactors; k++) {
				val = (Nuk.get(u, k) + alpha.get(k)) / (Nu.get(u) + sumAlpha);
				PukSum.add(u, k, val);
			}
		}

		for (int k = 0; k < numFactors; k++) {
			for (int i = 0; i < numWords; i++) {
				val = (Nki.get(k, i) + beta.get(i)) / (Nk.get(k) + sumBeta);
				PkiSum.add(k, i, val);
			}
		}
		numStats++;
	}

	protected void estimateParams() {
		Puk = PukSum.scale(1.0 / numStats);
		Pki = PkiSum.scale(1.0 / numStats);
	}

	
	public void saveModel() throws Exception {
		// make a folder
		String dirPath = FileIO.makeDirectory(outPath, "LDA");

		// suffix info
		String suffix = foldInfo + ".bin";

		// write matrices Theta, 
		FileIO.serialize(Puk, dirPath + "Puk" + suffix);
		FileIO.serialize(Pki, dirPath + "Pki" + suffix);

		Logs.debug("Learned models are saved to folder \"{}\"", dirPath);
	}
	
	public DenseMatrix[] loadModel() throws Exception {
		// make a folder
		String dirPath = FileIO.makeDirectory(outPath, "LDA");

		// suffix info
		String suffix = foldInfo + ".bin";

		// write matrices P, Q
		Puk = (DenseMatrix) FileIO.deserialize(dirPath + "Puk" + suffix);
		Pki = (DenseMatrix) FileIO.deserialize(dirPath + "Pki" + suffix);
		
		return new DenseMatrix[] {Puk, Pki};

	}
	
	

}