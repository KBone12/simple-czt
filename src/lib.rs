//! Provides functions to compute CZT/ICZT in O(nlogn)

use num_complex::Complex;
use num_traits::{
    cast::FromPrimitive,
    float::Float,
    identities::{One, Zero},
};
use rustfft::{FFTnum, FFTplanner};

/// `f32` or `f64` that can compute FFT
pub trait CZTnum: Copy + Float + FFTnum + Sync + Send {}

impl CZTnum for f32 {}
impl CZTnum for f64 {}

/// Computes CZT on the `input` buffer and places the result in the `output` buffer.
///
/// Do not use the `input` buffer because this function uses it as computing space.
pub fn czt<T: CZTnum>(
    input: &mut [Complex<T>],
    output: &mut [Complex<T>],
    w: Complex<T>,
    a: Complex<T>,
) {
    let mut r = vec![Complex::zero(); input.len()];
    for k in 0..input.len() {
        input[k] = w.powf(FromPrimitive::from_f64((k * k) as f64 / 2.0).unwrap())
            * a.powi(-(k as i32))
            * input[k];
        r[k] = w.powf(FromPrimitive::from_f64((k * k) as f64 / -2.0).unwrap());
    }
    let mut c = vec![Complex::zero(); output.len()];
    for k in 0..output.len() {
        c[k] = w.powf(FromPrimitive::from_f64((k * k) as f64 / -2.0).unwrap());
    }
    toeplitz_multiply_embedded(&r, &c, input, output);
    for k in 0..output.len() {
        output[k] = w.powf(FromPrimitive::from_f64((k * k) as f64 / 2.0).unwrap()) * output[k];
    }
}

/// Computes ICZT on the `input` buffer and places the result in the `output` buffer.
///
/// Do not use the `input` buffer because this function uses it as computing space.
pub fn iczt<T: CZTnum>(
    input: &mut [Complex<T>],
    output: &mut [Complex<T>],
    w: Complex<T>,
    a: Complex<T>,
) {
    assert_eq!(
        input.len(),
        output.len(),
        "The output length should equal to the input length."
    );
    for k in 0..output.len() {
        output[k] = w.powf(FromPrimitive::from_f64((k * k) as f64 / -2.0).unwrap()) * input[k];
    }
    let mut p: Vec<Complex<T>> = vec![Complex::zero(); output.len()];
    p[0] = Complex::one();
    for k in 1..p.len() {
        p[k] = p[k - 1] * (w.powf(FromPrimitive::from_usize(k).unwrap()) - Complex::one());
    }
    let mut u: Vec<Complex<T>> = vec![Complex::zero(); output.len()];
    for k in 0..u.len() as isize {
        u[k as usize] = (-Complex::<T>::one()).powu(k as u32)
            * w.powf(
                FromPrimitive::from_f64(
                    (2 * k * k - (2 * u.len() as isize - 1) * k
                        + u.len() as isize * (u.len() as isize - 1)) as f64
                        / 2.0,
                )
                .unwrap(),
            )
            / (p[u.len() - k as usize - 1] * p[k as usize]);
    }
    let z: Vec<Complex<T>> = vec![Complex::zero(); output.len()];
    let u_hat: Vec<_> = [
        vec![Complex::zero()],
        u.iter().skip(1).copied().rev().collect(),
    ]
    .concat()
    .to_vec();
    let mut u_tilde: Vec<Complex<T>> = vec![Complex::zero(); output.len()];
    u_tilde[0] = u[0];
    let mut x1 = vec![Complex::zero(); output.len()];
    let output_vec = output.to_vec();
    toeplitz_multiply_embedded(&u_hat, &z, &mut output_vec.clone(), &mut x1);
    let mut x2 = vec![Complex::zero(); output.len()];
    toeplitz_multiply_embedded(&z, &u_hat, &mut x1, &mut x2);
    let mut x3 = vec![Complex::zero(); output.len()];
    toeplitz_multiply_embedded(&u, &u_tilde, &mut output_vec.clone(), &mut x3);
    let mut x4 = vec![Complex::zero(); output.len()];
    toeplitz_multiply_embedded(&u_tilde, &u, &mut x3, &mut x4);
    for k in 0..output.len() {
        output[k] = (x4[k] - x2[k]) / u[0];
    }
    for k in 0..output.len() {
        output[k] = a.powu(k as u32)
            * w.powf(FromPrimitive::from_f64((k * k) as f64 / -2.0).unwrap())
            * output[k];
    }
}

fn toeplitz_multiply_embedded<T: CZTnum>(
    r: &[Complex<T>],
    c: &[Complex<T>],
    input: &[Complex<T>],
    output: &mut [Complex<T>],
) {
    let n = 2usize.pow(((input.len() + output.len() - 1) as f64).log2().ceil() as u32);
    let mut tmp = vec![Complex::zero(); n];
    for k in 0..c.len() {
        tmp[k] = c[k];
    }
    for k in 1..r.len() {
        tmp[n - k] = r[k];
    }
    let mut x = vec![Complex::zero(); n];
    for k in 0..input.len() {
        x[k] = input[k];
    }
    let mut y = vec![Complex::zero(); n];
    circulant_multiply(&mut tmp, &mut x, &mut y);
    for k in 0..output.len() {
        output[k] = y[k];
    }
}

fn circulant_multiply<T: CZTnum>(
    c: &mut [Complex<T>],
    x: &mut [Complex<T>],
    output: &mut [Complex<T>],
) {
    let mut c_out = vec![Complex::zero(); c.len()];
    let fft = FFTplanner::new(false).plan_fft(c.len());
    fft.process(c, &mut c_out);
    let mut x_out = vec![Complex::zero(); x.len()];
    fft.process(x, &mut x_out);
    let n = c_out.len();
    let mut y_in = c_out
        .into_iter()
        .zip(x_out.into_iter())
        .map(|(cc, xx)| cc * xx)
        .map(|cc| {
            Complex::new(
                cc.re / FromPrimitive::from_usize(n).unwrap(),
                cc.im / FromPrimitive::from_usize(n).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let ifft = FFTplanner::new(true).plan_fft(y_in.len());
    ifft.process(&mut y_in, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    use num_complex::Complex;
    use rustfft::{algorithm::DFT, FFT};

    #[test]
    fn czt_matches_dft() {
        // The `signal` is a 440Hz sinusoid on sample rate 48000Hz
        let n = 8192;
        let frequency = 440.0 / 48000.0;
        let signal = (0..n)
            .map(|s| (2.0 * std::f64::consts::PI * frequency * s as f64).sin())
            .map(|s| Complex::new(s, 0.0))
            .collect::<Vec<Complex<f64>>>();
        let mut dft_input = signal.clone();
        let mut czt_input = signal.clone();
        let dft = DFT::new(n, false);
        let mut dft_output = vec![Complex::zero(); n];
        let mut czt_output = vec![Complex::zero(); n];

        dft.process(&mut dft_input, &mut dft_output);
        czt(
            &mut czt_input,
            &mut czt_output,
            Complex::new(0.0, -2.0 * std::f64::consts::PI / n as f64).exp(),
            Complex::new(1.0, 0.0),
        );

        let error: f64 = dft_output
            .iter()
            .zip(czt_output.iter())
            .map(|(d, c)| (d - c).norm())
            .sum();

        assert!(error < 0.01);
    }

    #[test]
    fn czt_then_iczt_should_be_identity() {
        // The `signal` is a 440Hz sinusoid on sample rate 48000Hz
        let n = 8192;
        let frequency = 440.0 / 48000.0;
        let signal = (0..n)
            .map(|s| (2.0 * std::f64::consts::PI * frequency * s as f64).sin())
            .map(|s| Complex::new(s, 0.0))
            .collect::<Vec<Complex<f64>>>();
        let w = Complex::new(
            0.0,
            -2.0 * std::f64::consts::PI / n as f64 + 0.5 * std::f64::consts::PI,
        )
        .exp();
        let a = 1.0 * Complex::one();
        let mut czt_input = signal.clone();
        let mut czt_output = vec![Complex::zero(); n];
        let mut iczt_output = vec![Complex::zero(); n];

        czt(&mut czt_input, &mut czt_output, w, a);
        iczt(&mut czt_output, &mut iczt_output, w, a);

        let error: f64 = signal
            .iter()
            .zip(iczt_output.iter())
            .map(|(s, c)| (s - c).norm())
            .sum();

        assert!(error < 0.01, "Error: {}", error);
    }
}
