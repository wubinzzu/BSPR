// Copyright (C) 2014-2015 Guibing Guo
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

package librec.util;

import static java.lang.Math.exp;

public class Maths {

	/**
	 * Golden ratio: http://en.wikipedia.org/wiki/Golden_ratio
	 * 
	 * <p>
	 * (a+b)/a = a/b = phi (golden ratio) = 1.618033988749895
	 * </p>
	 */
	public final static double golden_ratio = 0.5 * (Math.sqrt(5) + 1);

	public final static double zero = 1e-6;

	public static boolean isEqual(double d1, double d2) {
		return Math.abs(d1 - d2) < zero;
	}

	/**
	 * Check if given string is a number (digits only)
	 * 
	 * @param string
	 * @return
	 */
	public static boolean isNumber(String string) {
		return string.matches("^\\d+$");
	}

	/**
	 * Check if given string is numeric (-+0..9(.)0...9)
	 * 
	 * @param string
	 * @return
	 */
	public static boolean isNumeric(String string) {
		return string.matches("^[-+]?\\d+(\\.\\d+)?$");
	}

	/**
	 * Check if given string is number with dot separator and two decimals.
	 * 
	 * @param string
	 * @return
	 */
	public static boolean isNumberWith2Decimals(String string) {
		return string.matches("^\\d+\\.\\d{2}$");
	}

	public static boolean isInt(double data) {
		return (Maths.isEqual(data, Math.floor(data))) && !Double.isInfinite(data);
	}

	/**
	 * @return n!
	 */
	public static int factorial(int n) {
		if (n < 0)
			return 0;
		if (n == 0 || n == 1)
			return 1;
		else
			return n * factorial(n - 1);
	}

	/**
	 * @return ln(e)=log_e(n)
	 */
	public static double ln(double n) {
		return Math.log(n);
	}
	public static double safelog(double x) {
		if (x <= 0)
			return (-10000);
		else
			return (Math.log(x));
	}
	public static double log(double n, int base) {
		return Math.log(n) / Math.log(base);
	}

	/**
	 * given log(a) and log(b), return log(a + b)
	 */
	public static double logSum(double log_a, double log_b) {
		double v;

		if (log_a < log_b) {
			v = log_b + Math.log(1 + exp(log_a - log_b));
		} else {
			v = log_a + Math.log(1 + exp(log_b - log_a));
		}
		return (v);
	}

	/**
	 * logistic function g(x)
	 */
	public static double logistic(double x) {
		return g(x);
	}

	/**
	 * logistic function g(x)
	 */
	public static double g(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	
	/**
	 * logistic function g(x)
	 * @throws Exception 
	 */
	public static double[] softmax(double[] x) throws Exception {
		double[] expx = new double[x.length];
		for (int i=0; i<x.length; i++){
			expx[i] = Math.exp(x[i]);
		}
		return norm(expx);
	}
    
	public static double[] norm(double[] x) throws Exception {
		double sm = sum(x);
		double[] result = new double[x.length];
		for (int i=0; i<x.length; i++){
			result[i] = x[i]/sm;
		}
		
		return result;
	}
	
	public static double sum(double[] x) {
		double sum = 0;
		for (int i=0; i<x.length; i++){
			sum += x[i];
		}
		return sum;
	}
	
	/**
	 * gradient value of logistic function g(x)
	 */
	public static double gd(double x) {
		return g(x) * g(-x);
	}

	/**
	 * get the normalized value using min-max normalizaiton
	 * 
	 * @param x
	 *            value to be normalized
	 * @param min
	 *            min value
	 * @param max
	 *            max value
	 * @return normalized value
	 */
	public static double normalize(double x, double min, double max) {
		if (max > min)
			return (x - min) / (max - min);
		else if (isEqual(min, max))
			return x / max;
		return x;
	}

	/**
	 * Fabonacci sequence
	 * 
	 */
	public static int fabonacci(int n) {
		assert n > 0;

		if (n == 1)
			return 0;
		else if (n == 2)
			return 1;
		else
			return fabonacci(n - 1) + fabonacci(n - 2);
	}

	/**
	 * greatest common divisor (gcd) or greatest common factor (gcf)
	 * 
	 * <p>
	 * reference: http://en.wikipedia.org/wiki/Greatest_common_divisor
	 * </p>
	 * 
	 */
	public static int gcd(int a, int b) {
		if (b == 0)
			return a;
		else
			return gcd(b, a % b);
	}

	/**
	 * least common multiple (lcm)
	 * 
	 */
	public static int lcm(int a, int b) {
		if (a > 0 && b > 0)
			return (int) ((0.0 + a * b) / gcd(a, b));
		else
			return 0;
	}

	/** sqrt(a^2 + b^2) without under/overflow. **/

	public static double hypot(double a, double b) {
		double r;
		if (Math.abs(a) > Math.abs(b)) {
			r = b / a;
			r = Math.abs(a) * Math.sqrt(1 + r * r);
		} else if (!isEqual(b, 0.0)) {
			r = a / b;
			r = Math.abs(b) * Math.sqrt(1 + r * r);
		} else {
			r = 0.0;
		}
		return r;
	}

}
