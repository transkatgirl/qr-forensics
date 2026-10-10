//! Getting a photo ready for the pipeline: upright and opaque. The size
//! it is worked on at is one of its adjustments
//! ([`super::adjust::Enlarge`]), which a small photo starts enlarged by,
//! so that its modules can be seen.

use std::path::Path;

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, ImageResult, RgbaImage};

use super::Luma;
use super::adjust::Adjust;

/// A photo from a file, as a viewer shows it: its format told by its
/// first bytes, whatever its name says, and turned upright as its EXIF
/// orientation asks. `image::open` goes by the name alone, and a PNG
/// saved as `photo.jpg`, as one of OpenCV's own test images is, would not
/// load; and a phone writes every photo as the sensor lies and notes the
/// turn beside it, so that a photo taken upright was shown, and its
/// outline counted, on its side. OpenCV's `imread` does both.
pub fn open(path: &Path) -> ImageResult<RgbaImage> {
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    // A photo whose orientation cannot be read is shown as it is stored.
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgba8())
}

/// Lay the photo on white paper wherever it is transparent, leaving it
/// opaque. An encoder that draws a symbol's dark modules on a transparent
/// background leaves the colour under the transparency at whatever it
/// was, most often black, and read without its alpha such a symbol is a
/// black rectangle; a browser or a viewer shows it on white, and so does
/// this.
pub fn flatten_alpha(image: &mut RgbaImage) {
    for pixel in image.pixels_mut() {
        let alpha = u32::from(pixel[3]);
        if alpha == 255 {
            continue;
        }
        for channel in &mut pixel.0[..3] {
            *channel = ((u32::from(*channel) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
        }
        pixel[3] = 255;
    }
}

/// The luminance the pipeline reads a photo by, under `adjust`: that of
/// the photo on white paper where it was transparent (see
/// [`flatten_alpha`]), at the size `adjust` has it. A photo as it is
/// first loaded is read under [`Adjust::for_photo`], which enlarges a
/// small one; the luminance then keeps that of the photo at its own size
/// beside it ([`Luma::unenlarged`]).
pub fn luma(mut image: RgbaImage, adjust: &Adjust) -> Luma {
    // Before any resampling, which would otherwise bleed the hidden
    // colour of transparent pixels into the edges of opaque ones.
    flatten_alpha(&mut image);
    adjust.apply(image.as_raw(), image.width() as usize, image.height() as usize).1
}

/// [`luma`] of a photo as it is first loaded: under no adjustment but
/// the enlargement a small one starts with.
pub fn luma_as_loaded(image: RgbaImage) -> Luma {
    let adjust = Adjust::for_photo(image.width(), image.height());
    luma(image, &adjust)
}

/// How much wider than itself, on each side, the box about an outline is
/// cut: room for a quiet zone, and for an outline that was drawn tight.
pub const CROP_MARGIN: f64 = 0.2;
/// The least a crop may measure on a side, in pixels.
const CROP_MIN_SIDE: u32 = 16;

/// The upright box about `outline`, a quadrilateral in the pixels of a
/// photo `width` by `height`, widened by [`CROP_MARGIN`] and kept inside
/// the photo: left, top, width and height. `None` when the outline lies
/// off the photo, or the box would be too small to hold a symbol.
pub fn crop_box(outline: [[f64; 2]; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    let (x0, x1) = outline.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
    let (y0, y1) = outline.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
    if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
        return None;
    }
    let (mx, my) = (CROP_MARGIN * (x1 - x0), CROP_MARGIN * (y1 - y0));
    let left = (x0 - mx).floor().clamp(0.0, f64::from(width)) as u32;
    let top = (y0 - my).floor().clamp(0.0, f64::from(height)) as u32;
    let right = (x1 + mx).ceil().clamp(0.0, f64::from(width)) as u32;
    let bottom = (y1 + my).ceil().clamp(0.0, f64::from(height)) as u32;
    (right >= left + CROP_MIN_SIDE && bottom >= top + CROP_MIN_SIDE).then_some([left, top, right - left, bottom - top])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_photo_is_enlarged_with_each_pixel_where_it_lay() {
        let tiny = RgbaImage::from_fn(11, 11, |x, y| {
            let v = if (x + y) % 2 == 0 { 0 } else { 255 };
            image::Rgba([v, v, v, 255])
        });
        let adjust = Adjust::for_photo(11, 11);
        let (big, read) = adjust.apply(tiny.as_raw(), 11, 11);
        assert_eq!((read.width, read.height), (11 * 24, 11 * 24));
        let big = RgbaImage::from_raw(11 * 24, 11 * 24, big).expect("the pixels are of the size the luminance has");
        // Each pixel keeps its value across its middle, and the edge
        // between two lies where it lay: either side of it is the
        // nearer pixel's.
        let level = |x: u32, y: u32| i32::from(big.get_pixel(x, y)[0]);
        for (x, y) in [(4, 7), (5, 5), (7, 4), (8, 8)] {
            let source = i32::from(tiny.get_pixel(x, y)[0]);
            assert!((level(24 * x + 12, 24 * y + 12) - source).abs() <= 2, "{x}, {y}: {}", level(24 * x + 12, 24 * y + 12));
        }
        assert!(level(5 * 24 - 1, 5 * 24 + 12) > 128 && level(5 * 24, 5 * 24 + 12) < 128);
        assert!(big.pixels().all(|pixel| pixel[1] == pixel[0] && pixel[2] == pixel[0] && pixel[3] == 255));
        // A photo that needs no enlarging is read at its size, however
        // large it is, and a thin one too long to enlarge as it is.
        let read = |width: u32, height: u32| {
            let read = luma_as_loaded(RgbaImage::new(width, height));
            (read.width, read.height, read.unenlarged.is_some())
        };
        assert_eq!(read(640, 480), (640, 480, false));
        assert_eq!(read(3000, 100), (3000, 100, false));
        // A long thin render is judged by its height.
        assert_eq!(read(587, 86), (587 * 3, 86 * 3, true));
    }

    #[test]
    fn a_small_photo_is_read_enlarged_with_its_own_pixels_beside_it() {
        // Opaque ink in squares on transparent black, 100 by 60.
        let small = RgbaImage::from_fn(100, 60, |x, y| image::Rgba([0, 0, 0, if (x / 10 + y / 10) % 2 == 0 { 255 } else { 0 }]));
        let read = luma_as_loaded(small.clone());
        assert_eq!((read.width, read.height), (500, 300));
        let own = read.unenlarged.as_ref().expect("a photo enlarged keeps its own pixels");
        assert_eq!((own.width, own.height), (100, 60));
        // Its own pixels, on white where they were transparent, each
        // ink or paper; enlarged, the edges between them are ramps.
        assert!((own.data[0] - 0.0).abs() < 1e-6 && (own.data[10] - 1.0).abs() < 1e-6);
        assert!(own.data.iter().all(|&v| !(1e-6..=1.0 - 1e-6).contains(&v)));
        assert!(read.data.iter().any(|&v| (0.3..=0.7).contains(&v)));
        // Under adjustments that leave it at its size, it is read at it.
        let plain = luma(small, &Adjust::default());
        assert_eq!((plain.width, plain.height, plain.unenlarged.is_none()), (100, 60, true));
        assert_eq!(plain.data, own.data);
        // A photo that needs no enlarging is read as it is.
        let fine = luma_as_loaded(RgbaImage::new(640, 480));
        assert_eq!((fine.width, fine.height, fine.unenlarged.is_none()), (640, 480, true));
        assert!(fine.data.iter().all(|&v| (v - 1.0).abs() < 1e-6), "transparent, so white");
    }

    #[test]
    fn a_crop_is_the_outline_with_room_about_it() {
        // An outline at an angle: the box is upright about its extremes,
        // a fifth of its size wider on every side.
        let outline = [[1000.0, 2000.0], [1500.0, 2100.0], [1400.0, 2600.0], [900.0, 2500.0]];
        assert_eq!(crop_box(outline, 8000, 6000), Some([780, 1880, 840, 840]));
        // Near the edge, the box stops at the photo's.
        let corner = [[10.0, 20.0], [510.0, 20.0], [510.0, 520.0], [10.0, 520.0]];
        assert_eq!(crop_box(corner, 600, 560), Some([0, 0, 600, 560]));
        // An outline off the photo, or one too small to hold a symbol,
        // gives nothing.
        let outside = outline.map(|[x, y]| [x + 9000.0, y]);
        assert_eq!(crop_box(outside, 8000, 6000), None);
        assert_eq!(crop_box([[5.0, 5.0], [9.0, 5.0], [9.0, 9.0], [5.0, 9.0]], 100, 100), None);
        assert_eq!(crop_box([[f64::NAN, 0.0]; 4], 100, 100), None);
    }

    #[test]
    fn a_photo_opens_by_its_bytes_and_stands_upright() {
        let dir = std::env::temp_dir();
        let name = |what: &str| dir.join(format!("qr-forensics-open-{}-{what}", std::process::id()));
        // Dark on the left, light on the right, twice as wide as tall.
        let photo = RgbaImage::from_fn(40, 20, |x, _| if x < 20 { image::Rgba([10, 10, 10, 255]) } else { image::Rgba([240, 240, 240, 255]) });

        // A PNG under a name that says JPEG.
        let misnamed = name("misnamed.jpg");
        photo.save_with_format(&misnamed, image::ImageFormat::Png).unwrap();
        assert!(image::open(&misnamed).is_err(), "the name alone misleads");
        assert_eq!(open(&misnamed).unwrap(), photo);

        // A JPEG whose EXIF block says the camera was held on its side:
        // orientation 6, to be turned a quarter clockwise for showing,
        // which brings the dark left half to the top.
        let mut jpeg = Vec::new();
        DynamicImage::ImageRgba8(photo.clone()).into_rgb8().write_to(&mut std::io::Cursor::new(&mut jpeg), image::ImageFormat::Jpeg).unwrap();
        let mut exif = b"Exif\0\0II*\0".to_vec();
        exif.extend(8u32.to_le_bytes());
        exif.extend(1u16.to_le_bytes());
        exif.extend([0x12, 0x01, 0x03, 0x00]);
        exif.extend(1u32.to_le_bytes());
        exif.extend([6, 0, 0, 0]);
        exif.extend(0u32.to_le_bytes());
        let mut turned = jpeg[..2].to_vec();
        turned.extend([0xFF, 0xE1]);
        turned.extend((exif.len() as u16 + 2).to_be_bytes());
        turned.extend(&exif);
        turned.extend(&jpeg[2..]);
        let sideways = name("sideways.jpg");
        std::fs::write(&sideways, &turned).unwrap();
        let upright = open(&sideways).unwrap();
        assert_eq!(upright.dimensions(), (20, 40));
        assert!(upright.get_pixel(10, 5)[0] < 60 && upright.get_pixel(10, 35)[0] > 200);

        let _ = std::fs::remove_file(misnamed);
        let _ = std::fs::remove_file(sideways);
    }

    #[test]
    fn transparency_shows_white_paper() {
        // Dark modules drawn opaque on a background that is transparent
        // black, as a canvas exports them; a half-transparent edge; and a
        // colour that is opaque already.
        let mut image = RgbaImage::from_fn(300, 300, |x, _| match x {
            0..100 => image::Rgba([0, 0, 0, 255]),
            100..200 => image::Rgba([0, 0, 0, 0]),
            200..250 => image::Rgba([0, 0, 0, 128]),
            _ => image::Rgba([10, 200, 30, 255]),
        });
        flatten_alpha(&mut image);
        assert_eq!(image.get_pixel(50, 7), &image::Rgba([0, 0, 0, 255]));
        assert_eq!(image.get_pixel(150, 7), &image::Rgba([255, 255, 255, 255]));
        assert_eq!(image.get_pixel(225, 7), &image::Rgba([127, 127, 127, 255]));
        assert_eq!(image.get_pixel(275, 7), &image::Rgba([10, 200, 30, 255]));
        // The luminance is read off the photo flattened, at any size:
        // enlarged, no hidden black bleeds into the white.
        let loaded = luma_as_loaded(RgbaImage::from_pixel(100, 100, image::Rgba([0, 0, 0, 0])));
        assert_eq!((loaded.width, loaded.height), (300, 300));
        assert!(loaded.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
