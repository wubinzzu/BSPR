// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.ranking;

import java.util.Date;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import librec.util.Randoms;
import librec.util.Strings;
/**
 * 
 * Rendle et al., <strong>BPR: Bayesian Personalized Ranking from Implicit Feedback</strong>, UAI 2009.
 * 
 * @author guoguibing
 * 
 */
public class BPR extends IterativeRecommender {

	public BPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
	    P.init(0,0.01);
        Q.init(0,0.01);
		userCache = trainMatrix.rowCache(cacheSpec);
	}

	@Override
	protected void buildModel() throws Exception {
		 int totalsize=trainMatrix.size();
			for (int iter = 1; iter <= numIters; iter++) {

				loss = 0;
				for (int s = 0, smax =totalsize; s < smax; s++) {
				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(numUsers);
					SparseVector pu = userCache.get(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = Randoms.uniform(numItems);
					} while (pu.contains(j));

					break;
				}

				// update parameters
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double xuij=xui-xuj;
				double vals = -Math.log(g(xuij));
				loss += vals;

				double cmg = g(-xuij);

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
//			if(iter>=1){	
//			System.out.println(iter+":  "+this.getEvalInfo(evalRankings()) + " " + new Date());
//			}
			System.out.println(iter+":  " + " " + new Date());
		}
	}
	@Override
	public String toString() {
		return Strings.toString(new Object[] {numFactors, initLRate,regU, regI, numIters }, ",");
	}
}
