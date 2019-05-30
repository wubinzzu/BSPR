package librec.ranking;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Date;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import librec.util.FileIO;
import librec.util.Randoms;
import librec.util.Strings;

@Configuration("epsilon1,epsilon2,beta,numFactors, initLRate, regU, regI, numIters")
public class BSPR extends IterativeRecommender {
    String relatedfile;
    private double maxloss;
    public SparseMatrix relatedmatrix;
    DenseMatrix Z;
    double beta;
    private float epsilon1,epsilon2;
	public BSPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws IOException {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true;
		initByNorm = false;
		 beta = algoOptions.getDouble("-beta");
		 relatedfile = cf.getPath("dataset.related");
		 relatedmatrix = this.getboughttogether();
		 epsilon1=algoOptions.getFloat("-epsilon1");
		 epsilon2=algoOptions.getFloat("-epsilon2");
		 
		 System.out.println("relations="+relatedmatrix.size());
		 
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		userCache = trainMatrix.rowCache(cacheSpec);
		 P.init(0, 0.01);
		 Q.init(0, 0.01);
		 Z=new DenseMatrix(numItems, numFactors);
	     Z.init(0, 0.01);  
	 	maxloss =1+0.5*(Math.floor(Math.log(numItems+1)/Math.log(2)-1));
	}
	@Override
	protected void buildModel() throws Exception {
		int ratings=trainMatrix.size();
  		int relations=relatedmatrix.size();
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
				double l_loss=1+0.5*(Math.floor(Math.log(l_rank+1)/Math.log(2)-1));
				
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
			
				
			
			for (int s = 0, smax =relations; s < smax; s++) {
				// randomly draw (i, j,K)
				int i = 0, j = 0, k = 0;
				int N=0;
				int Y=0;
				double xij=0;
				double xik=0;
				while (true) {
					i = Randoms.uniform(numItems);
				    
					SparseVector qi = relatedmatrix.row(i);
					 DenseVector	qif=Q.row(i);
					if (qi.getCount() == 0)
						continue;

					int[] js = qi.getIndex();
					j = js[Randoms.uniform(js.length)];
					 DenseVector	zjf=Z.row(j);
					 boolean con1 = false;
						N = 0;
						Y = 0;
						xij = qif.inner(zjf);
					do {
						N += 1;
						k = Randoms.uniform(numItems);
						DenseVector	zkf=Z.row(k);
						xik = qif.inner(zkf);
						con1 = xik > xij - epsilon2;
						Y = numItems - js.length;
						if (N > numItems - js.length - 1)
							break;
					} while (qi.contains(k) || !con1);

					break;
				}

				// update parameters
				double xijk=xij-xik;
				double cmg=0;
				double l_rank=Math.floor((Y-1)/N);
				double l_loss=1+0.5*(Math.floor(Math.log(l_rank+1)/Math.log(2)-1));
				l_loss/=maxloss;
				cmg = g(-xijk);
				cmg=cmg*l_loss;
				double vals = -Math.log(g(xijk));
				loss += vals;
				for (int f = 0; f < numFactors; f++) {
					double qif = Q.get(i, f);
					double zjf = Z.get(j, f);
					double zkf = Z.get(k, f);

					Q.add(i, f, lRate * (beta*cmg * (zjf - zkf) - regI * qif));
					Z.add(j, f, lRate * (beta*cmg * qif - regI * zjf));
					Z.add(k, f, lRate * (beta*cmg * (-qif) - regI * zkf));

				}
			}			
			if(iter>=1){	
			System.out.println(iter+":  "+this.getEvalInfo(evalRankings()) + " " + new Date());
			}
		}
	}
	
	public SparseMatrix getboughttogether() throws IOException {
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		Multimap<Integer, Integer> colMap = HashMultimap.create();
		BufferedReader br = FileIO.getReader(relatedfile);
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] itemrelations = line.split(",");
			String realitemid = itemrelations[0];
			if (rateDao.getItemIds().containsKey(realitemid)) {
				int inneritemid = rateDao.getItemIds().get(realitemid);
				for (int i = 1; i < itemrelations.length; i++) {
					if (rateDao.getItemIds().containsKey(itemrelations[i])) {
						int relatedinneriid = rateDao.getItemIds().get(itemrelations[i]);
						dataTable.put(inneritemid,relatedinneriid, 1.0);
						colMap.put(relatedinneriid,inneritemid);
					}
				}
			}
		}
		br.close();
		SparseMatrix itemrelatedmatrix = new SparseMatrix(numItems, numItems, dataTable, colMap);
		return itemrelatedmatrix;
	}  	
	@Override
	public String toString() {
		return Strings.toString(new Object[] {epsilon1,epsilon2,beta,numFactors, initLRate,regU, regI, numIters }, ",");
	}
}
