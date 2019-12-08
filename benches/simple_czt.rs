#![feature(test)]

use simple_czt::*;

extern crate test;
use test::Bencher;

use num_complex::Complex;
use num_traits::identities::{One, Zero};
use rustfft::FFTplanner;

fn bench_fft(n: usize, b: &mut Bencher) {
    // The `signal` is a 440Hz sinusoid on sample rate 48000Hz
    let frequency = 440.0 / 48000.0;
    let mut fft_input = (0..n)
        .map(|s| (2.0 * std::f64::consts::PI * frequency * s as f64).sin())
        .map(|s| Complex::new(s, 0.0))
        .collect::<Vec<Complex<f64>>>();
    let mut fft_output = vec![Complex::zero(); n];
    let fft = FFTplanner::new(false).plan_fft(n);
    b.iter(|| {
        fft.process(&mut fft_input, &mut fft_output);
    });
}

fn bench_czt(n: usize, b: &mut Bencher) {
    // The `signal` is a 440Hz sinusoid on sample rate 48000Hz
    let frequency = 440.0 / 48000.0;
    let w = Complex::new(
        0.0,
        -2.0 * std::f64::consts::PI / n as f64 + 0.5 * std::f64::consts::PI,
    )
    .exp();
    let a = 1.0 * Complex::one();
    let mut czt_input = (0..n)
        .map(|s| (2.0 * std::f64::consts::PI * frequency * s as f64).sin())
        .map(|s| Complex::new(s, 0.0))
        .collect::<Vec<Complex<f64>>>();
    let mut czt_output = vec![Complex::zero(); n];
    b.iter(|| {
        czt(&mut czt_input, &mut czt_output, w, a);
    });
}

fn bench_iczt(n: usize, b: &mut Bencher) {
    // The `signal` is a 440Hz sinusoid on sample rate 48000Hz
    let frequency = 440.0 / 48000.0;
    let w = Complex::new(
        0.0,
        -2.0 * std::f64::consts::PI / n as f64 + 0.5 * std::f64::consts::PI,
    )
    .exp();
    let a = 1.0 * Complex::one();
    let mut iczt_input = (0..n)
        .map(|s| (2.0 * std::f64::consts::PI * frequency * s as f64).sin())
        .map(|s| Complex::new(s, 0.0))
        .collect::<Vec<Complex<f64>>>();
    let mut iczt_output = vec![Complex::zero(); n];
    b.iter(|| {
        iczt(&mut iczt_input, &mut iczt_output, w, a);
    });
}

#[bench]
fn fft_n_10(b: &mut Bencher) {
    bench_fft(10, b);
}

#[bench]
fn fft_n_100(b: &mut Bencher) {
    bench_fft(100, b);
}

#[bench]
fn fft_n_1000(b: &mut Bencher) {
    bench_fft(1000, b);
}

#[bench]
fn fft_n_10000(b: &mut Bencher) {
    bench_fft(10000, b);
}

#[bench]
fn czt_n_10(b: &mut Bencher) {
    bench_czt(10, b);
}

#[bench]
fn czt_n_100(b: &mut Bencher) {
    bench_czt(100, b);
}

#[bench]
fn czt_n_1000(b: &mut Bencher) {
    bench_czt(1000, b);
}

#[bench]
fn czt_n_10000(b: &mut Bencher) {
    bench_czt(10000, b);
}

#[bench]
fn iczt_n_10(b: &mut Bencher) {
    bench_iczt(10, b);
}

#[bench]
fn iczt_n_100(b: &mut Bencher) {
    bench_iczt(100, b);
}

#[bench]
fn iczt_n_1000(b: &mut Bencher) {
    bench_iczt(1000, b);
}

#[bench]
fn iczt_n_10000(b: &mut Bencher) {
    bench_iczt(10000, b);
}
