package librec.ranking;

import java.util.HashMap;
import java.util.Map;

import librec.data.SparseMatrix;
import librec.intf.Recommender;

/**
 * Baseline: items are weighted by the number of ratings they received.
 * 
 * @author guoguibing
 * 
 */
public class MostPopular extends Recommender {

	private Map<Integer, Integer> itemPops;

	public MostPopular(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		// force to set as the ranking prediction method
		isRankingPred = true;
		setAlgoName("MostPop");
	}

	@Override
	protected void initModel() {
		itemPops = new HashMap<>();
	}

	@Override
	protected double ranking(int u, int j) {
		if (!itemPops.containsKey(j))
			itemPops.put(j, trainMatrix.columnSize(j));

		return itemPops.get(j);
	}

}
