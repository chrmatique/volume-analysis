use crate::data::models::{CorrelationMatrix, PcaResult};

/// Compute Pearson correlation between two equal-length slices
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }

    let mean_a = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b = b[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// Compute pairwise Pearson correlation matrix for multiple return series
pub fn compute_correlation_matrix(
    symbols: &[String],
    returns: &[Vec<f64>],
) -> CorrelationMatrix {
    let n = symbols.len();
    let mut matrix = vec![vec![0.0; n]; n];

    // Align all series to the same length (shortest)
    let min_len = returns.iter().map(|r| r.len()).min().unwrap_or(0);
    if min_len < 2 {
        return CorrelationMatrix {
            symbols: symbols.to_vec(),
            matrix,
        };
    }

    let aligned: Vec<&[f64]> = returns
        .iter()
        .map(|r| &r[r.len() - min_len..])
        .collect();

    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let corr = pearson_correlation(aligned[i], aligned[j]);
            matrix[i][j] = corr;
            matrix[j][i] = corr;
        }
    }

    CorrelationMatrix {
        symbols: symbols.to_vec(),
        matrix,
    }
}


/// Jacobi eigenvalue algorithm for real symmetric matrices.
///
/// On entry `a` is the symmetric matrix. On exit the off-diagonal elements are
/// near-zero and the diagonal holds the eigenvalues. The returned `Vec<Vec<f64>>`
/// is the accumulated rotation matrix whose columns are the eigenvectors.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[col]` is the
/// eigenvector corresponding to `eigenvalues[col]`.
fn jacobi_eigen(a: &mut Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    // Initialise V as the identity matrix (columns = eigenvectors)
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    let max_sweeps = 100;
    let tol = 1e-12_f64;

    for _ in 0..max_sweeps {
        // Find the largest off-diagonal element
        let mut max_val = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                }
            }
        }
        if max_val < tol {
            break;
        }

        // One full sweep over all off-diagonal pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < tol {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update the matrix entries
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * arp + c * arq;
                        a[q][r] = a[r][q];
                    }
                }

                // Accumulate rotation into V
                for r in 0..n {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigenvalues
        .iter()
        .enumerate()
        .for_each(|(i, _)| { let _ = i; }); // suppress unused warning

    (eigenvalues, v)
}

/// Perform PCA on a precomputed correlation matrix.
///
/// `t_obs` is the number of observations (rows of returns) used to build the
/// matrix; it is only needed to compute the Marchenko-Pastur upper bound.
/// Returns eigenvalues/vectors sorted in **descending** order.
pub fn compute_pca(symbols: &[String], matrix: &CorrelationMatrix, t_obs: usize) -> PcaResult {
    let n = symbols.len();
    if n == 0 {
        return PcaResult {
            eigenvalues: vec![],
            eigenvectors: vec![],
            symbols: vec![],
            mp_upper: 0.0,
        };
    }

    let mut a: Vec<Vec<f64>> = matrix.matrix.clone();
    let (eigenvalues, eigenvectors) = jacobi_eigen(&mut a);

    // Sort indices by descending eigenvalue
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_eigenvalues: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
    // eigenvectors stored as [pc_idx][sector_idx] (transpose of V columns)
    let sorted_eigenvectors: Vec<Vec<f64>> = order
        .iter()
        .map(|&col| (0..n).map(|row| eigenvectors[row][col]).collect())
        .collect();

    let q = if t_obs > 0 { n as f64 / t_obs as f64 } else { 0.0 };
    let mp_upper = (1.0 + q.sqrt()).powi(2);

    PcaResult {
        eigenvalues: sorted_eigenvalues,
        eigenvectors: sorted_eigenvectors,
        symbols: symbols.to_vec(),
        mp_upper,
    }
}

/// Compute a windowed correlation matrix using only the last `window` observations.
pub fn compute_correlation_matrix_windowed(
    symbols: &[String],
    returns: &[Vec<f64>],
    window: usize,
) -> CorrelationMatrix {
    let windowed: Vec<Vec<f64>> = returns
        .iter()
        .map(|r| {
            let start = r.len().saturating_sub(window);
            r[start..].to_vec()
        })
        .collect();
    compute_correlation_matrix(symbols, &windowed)
}

/// Compute average cross-sector correlation from a correlation matrix
pub fn average_cross_correlation(matrix: &CorrelationMatrix) -> f64 {
    let n = matrix.symbols.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += matrix.matrix[i][j];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr = pearson_correlation(&a, &b);
        assert!((corr + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix_diagonal() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let returns = vec![
            vec![0.01, -0.02, 0.03, 0.01, -0.01],
            vec![0.02, -0.01, 0.02, 0.015, -0.005],
            vec![-0.01, 0.03, -0.02, 0.005, 0.01],
        ];
        let cm = compute_correlation_matrix(&symbols, &returns);
        for i in 0..3 {
            assert!((cm.matrix[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_correlation_matrix_symmetric() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![
            vec![0.01, -0.02, 0.03, 0.01],
            vec![0.02, -0.01, 0.02, 0.015],
        ];
        let cm = compute_correlation_matrix(&symbols, &returns);
        assert!((cm.matrix[0][1] - cm.matrix[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_average_cross_correlation() {
        let cm = CorrelationMatrix {
            symbols: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            matrix: vec![
                vec![1.0, 0.8, 0.6],
                vec![0.8, 1.0, 0.7],
                vec![0.6, 0.7, 1.0],
            ],
        };
        let avg = average_cross_correlation(&cm);
        let expected = (0.8 + 0.6 + 0.7) / 3.0;
        assert!((avg - expected).abs() < 1e-10);
    }

    // --- Jacobi / PCA tests ---

    #[test]
    fn test_jacobi_identity_3x3() {
        let mut a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (eigenvalues, _) = jacobi_eigen(&mut a);
        for &ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10, "Expected eigenvalue 1.0, got {ev}");
        }
    }

    #[test]
    fn test_jacobi_2x2_known() {
        // [[1, 0.5], [0.5, 1]] has eigenvalues 1.5 and 0.5
        let mut a = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let (mut eigenvalues, _) = jacobi_eigen(&mut a);
        eigenvalues.sort_by(|x, y| y.partial_cmp(x).unwrap());
        assert!((eigenvalues[0] - 1.5).abs() < 1e-8, "Expected 1.5, got {}", eigenvalues[0]);
        assert!((eigenvalues[1] - 0.5).abs() < 1e-8, "Expected 0.5, got {}", eigenvalues[1]);
    }

    #[test]
    fn test_compute_pca_eigenvalues_sorted_descending() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let cm = CorrelationMatrix {
            symbols: symbols.clone(),
            matrix: vec![
                vec![1.0, 0.8, 0.6],
                vec![0.8, 1.0, 0.7],
                vec![0.6, 0.7, 1.0],
            ],
        };
        let pca = compute_pca(&symbols, &cm, 252);
        for i in 0..pca.eigenvalues.len().saturating_sub(1) {
            assert!(
                pca.eigenvalues[i] >= pca.eigenvalues[i + 1],
                "Eigenvalues not sorted descending at index {i}"
            );
        }
    }

    #[test]
    fn test_mp_upper_bound() {
        // n=11 sectors, t=252 trading days => mp_upper = (1 + sqrt(11/252))^2
        let symbols: Vec<String> = (0..11).map(|i| format!("S{i}")).collect();
        let identity: Vec<Vec<f64>> = (0..11)
            .map(|i| (0..11).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let cm = CorrelationMatrix { symbols: symbols.clone(), matrix: identity };
        let pca = compute_pca(&symbols, &cm, 252);
        let expected = (1.0 + (11.0_f64 / 252.0).sqrt()).powi(2);
        assert!(
            (pca.mp_upper - expected).abs() < 1e-10,
            "MP upper: expected {expected}, got {}",
            pca.mp_upper
        );
    }

    #[test]
    fn test_windowed_correlation_length() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        // 100 observations each
        let returns = vec![
            (0..100).map(|i| (i as f64 * 0.01).sin()).collect::<Vec<f64>>(),
            (0..100).map(|i| (i as f64 * 0.01).cos()).collect::<Vec<f64>>(),
        ];
        let window = 30;
        // The windowed correlation should be computed on the last 30 rows;
        // the matrix shape should still be 2×2
        let cm = compute_correlation_matrix_windowed(&symbols, &returns, window);
        assert_eq!(cm.matrix.len(), 2);
        assert_eq!(cm.matrix[0].len(), 2);
        assert!((cm.matrix[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_correlation_shorter_than_window() {
        // When the series is shorter than the window, all observations are used
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![
            vec![0.01, -0.02, 0.03],
            vec![0.02, -0.01, 0.02],
        ];
        let cm_full = compute_correlation_matrix(&symbols, &returns);
        let cm_windowed = compute_correlation_matrix_windowed(&symbols, &returns, 100);
        assert!((cm_full.matrix[0][1] - cm_windowed.matrix[0][1]).abs() < 1e-10);
    }
}
