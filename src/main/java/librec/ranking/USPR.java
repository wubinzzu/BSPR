package librec.ranking;
import java.io.IOException;
import java.util.Date;
import librec.data.Configuration;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import librec.util.Randoms;
import librec.util.Strings;

@Configuration("epsilon1,numFactors, initLRate, regU, regI, numIters")
public class USPR extends IterativeRecommender {
    private double maxloss;
    private float epsilon1;
	public USPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws IOException {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
		 epsilon1=algoOptions.getFloat("-epsilon1");
		 
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		userCache = trainMatrix.rowCache(cacheSpec);
		 P.init(0, 0.01);
		 Q.init(0, 0.01);  
			maxloss =1+0.5*(Math.floor(Math.log(numItems+1)/Math.log(2)-1));
	}
	@Override
	protected void buildModel() throws Exception {
		int ratings=trainMatrix.size();
		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			for (int s = 0, smax =ratings; s < smax; s++) {
				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;
				int N=0;
				int Y=0;
				double xui=0;
				double xuj=0;
				while (true) {
					u = Randoms.uniform(numUsers);
					SparseVector pu = trainMatrix.row(u);
				    DenseVector	puf=P.row(u);
					if (pu.getCount() == 0)
						continue;
					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];
					 DenseVector	qif=Q.row(i);
					boolean con1 = false;
					N = 0;
					Y = 0;
					xui = puf.inner(qif);
					do {
						N += 1;
						j = Randoms.uniform(numItems);
						DenseVector	qjf=Q.row(j);
						xuj = puf.inner(qjf);
						con1 = xuj > xui - epsilon1;
						Y = numItems - is.length;
						if (N > numItems - is.length - 1)
							break;
					} while (pu.contains(j) || !con1);

					break;
				}
				double xuij=xui-xuj;
				double cmg=0;
				double l_rank=Math.floor((Y-1)/N);
				double l_loss=0;
			
				l_loss =1+0.5*(Math.floor(Math.log(l_rank+1)/Math.log(2)-1));
				l_loss/=maxloss;
				cmg= g(-xuij);
				
				cmg=cmg*l_loss;
				double vals = -Math.log(g(xuij));//Note I use logistic loss as a surrogate for other losses
				loss += vals;

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);

					P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg * puf - regI * qif));
					Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));

					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
			}			
			//if(iter>=1){	
			//System.out.println(iter+":  "+this.getEvalInfo(evalRankings()) + " " + new Date());
			//}
			System.out.println(iter+":  "+ " " + new Date());
		}
	}
	@Override
	public String toString() {
		return Strings.toString(new Object[] {epsilon1,numFactors, initLRate,regU, regI, numIters }, ",");
	}
}
