extern crate nalgebra as na;

use na::{convert, DMatrix, DVector};
use std::time::{SystemTime};


fn select_indices(values: &DVector<f64>,
                  indices: &DVector<usize>) -> DVector<f64> {
    indices.map(|i| values[i])
}


fn tricubic(x: &DVector<f64>) -> DVector<f64> {
    x.map(|v|
        if (-1.0..=1.0).contains(&v) {
            (1.0 - v.abs().powi(3)).powi(3)
        } else {
            0.0
        }
    )
}


fn normalize(x: &DVector<f64>) -> (DVector<f64>, f64, f64) {
    let min_val = x.min();
    let max_val = x.max();
    let y = x.map(|value| (value - min_val) / (max_val - min_val));
    (y, min_val, max_val)
}


fn get_min_range(distances: &DVector<f64>, window: usize) -> DVector<usize> {
    let (min_idx, _) = distances.argmin();
    let n = distances.len();
    let range: DVector<usize>;

    if min_idx == 0 {
        range = DVector::<usize>::from_iterator(window, 0..window);
    } else if min_idx == n - 1 {
        range = DVector::<usize>::from_iterator(window, n - window..n);
    } else {
        let mut min_range: Vec<usize> = vec![min_idx];
        let mut l: usize = 1;
        while l < window {
            let i0: usize = min_range[0];
            let i1: usize = min_range[l-1];

            if i0 == 0 {
                min_range.push(i1 + 1);
            } else if (i1 == n - 1) || (distances[i0 - 1] < distances[i1 + 1]) {
                min_range.insert(0, i0 - 1);
            } else {
                min_range.push(i1 + 1);
            }
            l += 1;
        }
        range = DVector::<usize>::from_vec(min_range);
    }
    range
}


fn get_weights(distances: &DVector<f64>, min_range: &DVector<usize>) -> DVector<f64> {
    let selection = select_indices(distances, min_range);
    let max_distance = selection.max();
    let norm_distances = selection / max_distance;
    tricubic(&norm_distances)
}


pub struct Loess {
    xx: DVector<f64>,
    yy: DVector<f64>,
    min_y: f64,
    max_y: f64,
    min_x: f64,
    max_x: f64,
}


impl Loess {
    fn new(xs: &DVector<f64>, ys: &DVector<f64>) -> Self {
        let (norm_xs, min_x, max_x) = normalize(xs);
        let (norm_ys, min_y, max_y) = normalize(ys);
        Loess {
            xx: norm_xs,
            yy: norm_ys,
            min_y,
            max_y,
            min_x,
            max_x,
        }
    }

    fn normalize_x(&self, x: f64) -> f64 {
        (x - self.min_x) / (self.max_x - self.min_x)
    }

    fn denormalize_y(&self, y: f64) -> f64 {
        y * (self.max_y - self.min_y) + self.min_y
    }
    
    fn estimate_algebra(&self, 
                        n_x: f64,
                        window: usize,
                        min_range: &DVector<usize>,
                        weights: &DVector<f64>,
                        degree: usize) -> f64 {
        let wm = DMatrix::<f64>::from_diagonal(weights);
        let mut xm = DMatrix::<f64>::from_element(window, degree + 1, 1.0);
        let xp = DVector::<f64>::from_iterator(degree + 1, (0..=degree)
            .map(|p| n_x.powi(p as i32)));
        for i in 1..=degree {
            for j in 0..window {
                xm[(j, i)] = self.xx[min_range[j]].powi(i as i32);
            }
        }
        let ym: DMatrix<f64> = convert(select_indices(&self.yy, min_range));
        let xmt_wm = xm.transpose() * wm;
        let wmt_wm_xm = &xmt_wm * xm;
        let inv = wmt_wm_xm.pseudo_inverse(1e-5).unwrap();
        let beta = (inv * xmt_wm) * ym;
        let y = (beta.transpose() * xp)[0];
        y
    }
    
    fn estimate_stats(&self,
                      n_x: f64,
                      min_range: &DVector<usize>,
                      weights: &DVector<f64>
                     ) -> f64 {
        let xx = select_indices(&self.xx, min_range);
        let yy = select_indices(&self.yy, min_range);
        let sum_weight = weights.sum();
        let sum_weight_x = xx.dot(weights);
        let sum_weight_y = yy.dot(weights);
        let sum_weight_x2 = xx.component_mul(&xx).dot(weights);
        let sum_weight_xy = xx.component_mul(&yy).dot(weights);

        let mean_x = sum_weight_x / sum_weight;
        let mean_y = sum_weight_y / sum_weight;

        let b = (sum_weight_xy - mean_x * mean_y * sum_weight) /
            (sum_weight_x2 - mean_x * mean_x * sum_weight);
        let a = mean_y - b * mean_x;
        let y = a + b * n_x;
        y
    }

    fn estimate(&self, x: f64,
                window: usize,
                use_matrix: bool,
                degree: usize
               ) -> f64 {
        let n_x= self.normalize_x(x);
        let distances: DVector<f64> = (&self.xx - DVector::<f64>::repeat(self.xx.len(), n_x)).abs();
        let min_range: DVector<usize> = get_min_range(&distances, window);
        let weights: DVector<f64> = get_weights(&distances, &min_range);

        let y: f64 = if use_matrix || degree > 1 {
            self.estimate_algebra(n_x, window, &min_range, &weights, degree)
        } else {
            self.estimate_stats(n_x, &min_range, &weights)
        };

        self.denormalize_y(y)
    }
}


fn main() {
    let xx: DVector<f64> = DVector::<f64>::from_vec(
        vec![0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
                  4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
                  8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
                  14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
                  18.7572812]);

    let yy: DVector<f64> = DVector::<f64>::from_vec(
        vec![18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
                  213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
                  227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
                  160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
                  243.18828]);

    let loess: Loess = Loess::new(&xx, &yy);

    let start = SystemTime::now();
    for &x in xx.iter() {
        let y = loess.estimate(x,7,true,1);
        // println!("{}, {}", x, y);
    }
    let end = SystemTime::now();
    let elapsed = end.duration_since(start).expect("Time went backwards");
    println!("Elapsed time: {:?}", elapsed);
}
