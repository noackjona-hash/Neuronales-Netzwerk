//! Mathematical primitives, 2D vector operations, line segment intersection,
//! raycasting, and linear algebra built from scratch with zero external dependencies.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// 2D Vector with high-performance mathematical operations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const ONE: Vec2 = Vec2 { x: 1.0, y: 1.0 };
    pub const UNIT_X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const UNIT_Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Construct a unit vector from an angle in radians.
    #[inline(always)]
    pub fn from_angle(angle_rad: f32) -> Self {
        Self {
            x: angle_rad.cos(),
            y: angle_rad.sin(),
        }
    }

    /// Compute angle in radians: [-PI, PI].
    #[inline(always)]
    pub fn to_angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Dot product: x1 * x2 + y1 * y2.
    #[inline(always)]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// 2D Cross product (perpendicular dot product / determinant): x1 * y2 - y1 * x2.
    #[inline(always)]
    pub fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Squared Euclidean length.
    #[inline(always)]
    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length.
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Normalized unit vector. Returns Vec2::ZERO if length is near zero.
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-6 {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            Self::ZERO
        }
    }

    /// Squared distance to another point.
    #[inline(always)]
    pub fn distance_sq(self, other: Self) -> f32 {
        (self - other).length_sq()
    }

    /// Euclidean distance to another point.
    #[inline(always)]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    /// Rotate vector around the origin by `angle_rad` radians.
    #[inline(always)]
    pub fn rotate(self, angle_rad: f32) -> Self {
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        Self {
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
        }
    }

    /// Rotate vector around an arbitrary pivot point.
    #[inline(always)]
    pub fn rotate_around(self, pivot: Self, angle_rad: f32) -> Self {
        (self - pivot).rotate(angle_rad) + pivot
    }

    /// Perpendicular normal vector (rotated 90 degrees counter-clockwise).
    #[inline(always)]
    pub fn perpendicular(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// Linear interpolation between two vectors.
    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }

    /// Clamp vector magnitude to a maximum length.
    #[inline(always)]
    pub fn clamp_length(self, max_len: f32) -> Self {
        let len_sq = self.length_sq();
        if len_sq > max_len * max_len && len_sq > 0.0 {
            self * (max_len / len_sq.sqrt())
        } else {
            self
        }
    }

    /// Element-wise min.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Element-wise max.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }
}

// Operator overloads for Vec2
impl Add for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl Mul<Vec2> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl DivAssign<f32> for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

/// 2D Line Segment between `start` and `end`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineSegment {
    pub start: Vec2,
    pub end: Vec2,
}

impl LineSegment {
    #[inline(always)]
    pub const fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end }
    }

    /// Length of the segment.
    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.start.distance(self.end)
    }

    /// Normalized direction from `start` to `end`.
    #[inline(always)]
    pub fn direction(&self) -> Vec2 {
        (self.end - self.start).normalize()
    }

    /// Outward perpendicular normal.
    #[inline(always)]
    pub fn normal(&self) -> Vec2 {
        self.direction().perpendicular()
    }

    /// Midpoint of the line segment.
    #[inline(always)]
    pub fn midpoint(&self) -> Vec2 {
        (self.start + self.end) * 0.5
    }

    /// Computes the projection parameter t ∈ [0, 1] of a point onto this segment.
    pub fn closest_point_t(&self, point: Vec2) -> f32 {
        let seg_vec = self.end - self.start;
        let len_sq = seg_vec.length_sq();
        if len_sq < 1e-6 {
            return 0.0;
        }
        let pt_vec = point - self.start;
        (pt_vec.dot(seg_vec) / len_sq).clamp(0.0, 1.0)
    }

    /// Closest point on the segment to `point`.
    pub fn closest_point(&self, point: Vec2) -> Vec2 {
        let t = self.closest_point_t(point);
        self.start + (self.end - self.start) * t
    }

    /// Distance from `point` to this segment.
    pub fn distance_to_point(&self, point: Vec2) -> f32 {
        self.closest_point(point).distance(point)
    }

    /// Test intersection with another line segment.
    /// Returns the exact intersection point if segments intersect.
    pub fn intersect_segment(&self, other: &LineSegment) -> Option<Vec2> {
        let p = self.start;
        let r = self.end - self.start;
        let q = other.start;
        let s = other.end - other.start;

        let r_cross_s = r.cross(s);
        let q_minus_p = q - p;

        if r_cross_s.abs() < 1e-6 {
            // Parallel or collinear
            return None;
        }

        let t = q_minus_p.cross(s) / r_cross_s;
        let u = q_minus_p.cross(r) / r_cross_s;

        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            Some(p + r * t)
        } else {
            None
        }
    }
}

/// Raycast hit information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastHit {
    pub point: Vec2,
    pub distance: f32,
    pub fraction: f32, // Normalized distance [0.0, 1.0] where 1.0 = max_dist
    pub normal: Vec2,
}

/// 2D Ray starting at `origin` traveling along `direction` up to `max_dist`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray2 {
    pub origin: Vec2,
    pub direction: Vec2, // Must be normalized
    pub max_dist: f32,
}

impl Ray2 {
    #[inline(always)]
    pub fn new(origin: Vec2, direction: Vec2, max_dist: f32) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            max_dist,
        }
    }

    /// Cast ray against a line segment.
    pub fn cast_segment(&self, segment: &LineSegment) -> Option<RaycastHit> {
        let seg_vec = segment.end - segment.start;
        let denom = self.direction.cross(seg_vec);

        if denom.abs() < 1e-6 {
            // Ray and segment are parallel
            return None;
        }

        let orig_to_start = segment.start - self.origin;
        let t = orig_to_start.cross(seg_vec) / denom;
        let u = orig_to_start.cross(self.direction) / denom;

        if t >= 0.0 && t <= self.max_dist && (0.0..=1.0).contains(&u) {
            let hit_point = self.origin + self.direction * t;
            let normal = segment.normal();
            Some(RaycastHit {
                point: hit_point,
                distance: t,
                fraction: (t / self.max_dist).clamp(0.0, 1.0),
                normal,
            })
        } else {
            None
        }
    }

    /// Cast ray against a collection of line segments, returning the closest hit.
    pub fn cast_segments(&self, segments: &[LineSegment]) -> Option<RaycastHit> {
        let mut closest_hit: Option<RaycastHit> = None;

        for seg in segments {
            if let Some(hit) = self.cast_segment(seg) {
                match &closest_hit {
                    Some(current) => {
                        if hit.distance < current.distance {
                            closest_hit = Some(hit);
                        }
                    }
                    None => {
                        closest_hit = Some(hit);
                    }
                }
            }
        }

        closest_hit
    }
}

/// 2D Dynamic Matrix for neural network computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    /// Create a zero-initialized matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create matrix from elements generator.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> f32,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }

    /// Compute matrix-vector product: y = W * x (where W is rows x cols, x is cols x 1).
    pub fn dot_vector(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            self.cols,
            x.len(),
            "Matrix cols ({}) must match vector len ({})",
            self.cols,
            x.len()
        );
        let mut result = vec![0.0; self.rows];
        for r in 0..self.rows {
            let row_offset = r * self.cols;
            let mut sum = 0.0f32;
            for c in 0..self.cols {
                sum += self.data[row_offset + c] * x[c];
            }
            result[r] = sum;
        }
        result
    }

    /// Fast matrix-vector product adding directly to pre-allocated slice with bias: y[r] = sum(W[r,c]*x[c]) + b[r].
    pub fn forward_linear(&self, x: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(self.cols, x.len());
        assert_eq!(self.rows, b.len());
        assert_eq!(self.rows, out.len());

        for r in 0..self.rows {
            let row_offset = r * self.cols;
            let mut sum = b[r];
            for c in 0..self.cols {
                sum += self.data[row_offset + c] * x[c];
            }
            out[r] = sum;
        }
    }
}

/// Gaussian random number generator using Box-Muller transform from scratch.
pub struct GaussianRng;

impl GaussianRng {
    /// Generate standard normal random sample N(0, 1) using Box-Muller.
    pub fn sample_standard<R: rand::Rng + ?Sized>(rng: &mut R) -> f32 {
        let u1: f32 = rng.gen::<f32>().max(1e-7);
        let u2: f32 = rng.gen::<f32>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Generate normal sample N(mean, std_dev^2).
    pub fn sample<R: rand::Rng + ?Sized>(mean: f32, std_dev: f32, rng: &mut R) -> f32 {
        mean + std_dev * Self::sample_standard(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_math() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.dot(Vec2::new(2.0, 1.0)), 10.0);
        assert_eq!(a.cross(Vec2::new(1.0, 2.0)), 3.0 * 2.0 - 4.0 * 1.0);

        let rot = Vec2::UNIT_X.rotate(std::f32::consts::FRAC_PI_2);
        assert!((rot.x - 0.0).abs() < 1e-5);
        assert!((rot.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_segment_intersection() {
        let s1 = LineSegment::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let s2 = LineSegment::new(Vec2::new(0.0, 10.0), Vec2::new(10.0, 0.0));
        let hit = s1.intersect_segment(&s2).expect("Should intersect");
        assert!((hit.x - 5.0).abs() < 1e-5);
        assert!((hit.y - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_raycast_line() {
        let ray = Ray2::new(Vec2::new(5.0, 0.0), Vec2::new(0.0, 1.0), 100.0);
        let seg = LineSegment::new(Vec2::new(0.0, 20.0), Vec2::new(10.0, 20.0));
        let hit = ray.cast_segment(&seg).expect("Ray should hit line");
        assert!((hit.point.x - 5.0).abs() < 1e-5);
        assert!((hit.point.y - 20.0).abs() < 1e-5);
        assert!((hit.distance - 20.0).abs() < 1e-5);
        assert!((hit.fraction - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_vector() {
        let m = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let x = vec![1.0, 2.0, 3.0];
        let y = m.dot_vector(&x);
        assert_eq!(y, vec![14.0, 32.0]);
    }
}
