//! Getting a photo ready for the pipeline: upright, opaque, and large
//! enough for its modules to be seen. It is never made smaller.

use std::path::Path;

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, ImageResult, RgbaImage};

use super::Luma;
use super::adjust::Adjust;
use super::resize::resize;

/// The longest a small photo's longer side is enlarged to: see
/// [`MIN_PHOTO_SIDE`].
pub const MAX_ENLARGED_SIDE: u32 = 4096;
/// Photos smaller than this on their shorter side are scaled up by a
/// whole factor, so that a symbol rendered at a few pixels a module gets
/// modules the finder scan can see. A long thin rMQR render is judged by its
/// height. zxing-cpp has a separate path for such "pure" images; here they
/// are simply made bigger.
pub const MIN_PHOTO_SIDE: u32 = 256;

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

/// The photo as the pipeline works on it: on white paper where it was
/// transparent (see [`flatten_alpha`]), and, when it is small, enlarged
/// by a whole factor with Magic Kernel Sharp 2013 ([`super::resize`]),
/// as far as the longer side allows. Cubic and linear enlargement were
/// tried before it and lost more of the corpus than they gained: a
/// cubic rings at every edge, and even a linear ramp moves the corners
/// the edge tracer finds on crumpled paper. This kernel reads the
/// corpus as enlarging pixel for pixel did. It rounds the square
/// corners [`super::pure`] reads a rendered symbol off its pixels by,
/// so a render is read off the pixels it had before it was enlarged,
/// which [`luma`] keeps beside those it has after.
///
/// A large photo is left at its size. It was once reduced to 4096 pixels
/// on its longer side, and a small symbol in a large photo lost to that
/// the pixels its modules had. Only the search of the whole photo still
/// looks at a copy that size ([`super::detect::SEARCH_SIDE`]): what it
/// finds, like an outline drawn by hand, is refined and read in the
/// photo's own pixels.
pub fn normalize_size(mut image: RgbaImage) -> RgbaImage {
    // Before any resampling, which would otherwise bleed the hidden
    // colour of transparent pixels into the edges of opaque ones.
    flatten_alpha(&mut image);
    enlarged(&image).unwrap_or(image)
}

/// The luminance the pipeline reads a photo by, under `adjust`: that of
/// the photo as [`normalize_size`] leaves it, and, where that enlarged
/// it, with the luminance of the photo at its own size kept beside it
/// ([`Luma::unenlarged`]).
pub fn luma(mut image: RgbaImage, adjust: &Adjust) -> Luma {
    flatten_alpha(&mut image);
    match enlarged(&image) {
        Some(larger) => adjust.apply_photo(&larger, Some(&image)).1,
        None => adjust.apply_photo(&image, None).1,
    }
}

/// An opaque photo enlarged as [`normalize_size`] enlarges a small one;
/// `None` for a photo that needs none of it, which is every photo of
/// [`MIN_PHOTO_SIDE`] or more on its shorter side.
pub fn enlarged(image: &RgbaImage) -> Option<RgbaImage> {
    let longest = image.width().max(image.height()).max(1);
    let shortest = image.width().min(image.height()).max(1);
    if shortest >= MIN_PHOTO_SIDE {
        return None;
    }
    let factor = MIN_PHOTO_SIDE.div_ceil(shortest).min(MAX_ENLARGED_SIDE / longest).max(1);
    (factor > 1).then(|| {
        let (width, height) = (image.width() * factor, image.height() * factor);
        let pixels = resize(image.as_raw(), image.width() as usize, image.height() as usize, 4, width as usize, height as usize);
        RgbaImage::from_raw(width, height, pixels).expect("the pixels are of the size asked for")
    })
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
    fn sizes_are_brought_into_range() {
        let tiny = RgbaImage::from_fn(11, 11, |x, y| {
            let v = if (x + y) % 2 == 0 { 0 } else { 255 };
            image::Rgba([v, v, v, 255])
        });
        let big = normalize_size(tiny.clone());
        assert_eq!((big.width(), big.height()), (11 * 24, 11 * 24));
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
        let fine = RgbaImage::new(640, 480);
        assert_eq!(normalize_size(fine.clone()).dimensions(), (640, 480));
        // A large photo keeps every pixel it has.
        let huge = RgbaImage::new(8192, 4096);
        assert_eq!(normalize_size(huge).dimensions(), (8192, 4096));
        // A long thin render is judged by its height, within the limit on
        // its width.
        let strip = RgbaImage::new(587, 86);
        assert_eq!(normalize_size(strip).dimensions(), (587 * 3, 86 * 3));
        let long_strip = RgbaImage::new(3000, 100);
        assert_eq!(normalize_size(long_strip).dimensions(), (3000, 100), "no whole factor fits");
        // A photo that needs no enlarging is not copied to be told so,
        // however large it is, and a thin one too long to enlarge is
        // left as it is.
        assert!(enlarged(&fine).is_none());
        assert!(enlarged(&RgbaImage::new(8192, 4096)).is_none());
        assert!(enlarged(&RgbaImage::new(5000, 100)).is_none());
        assert_eq!(enlarged(&RgbaImage::new(100, 60)).map(|i| i.dimensions()), Some((500, 300)));
    }

    #[test]
    fn a_small_photo_is_read_enlarged_with_its_own_pixels_beside_it() {
        // Opaque ink in squares on transparent black, 100 by 60.
        let small = RgbaImage::from_fn(100, 60, |x, y| image::Rgba([0, 0, 0, if (x / 10 + y / 10) % 2 == 0 { 255 } else { 0 }]));
        let read = luma(small, &Adjust::default());
        assert_eq!((read.width, read.height), (500, 300));
        let own = read.unenlarged.as_ref().expect("a photo enlarged keeps its own pixels");
        assert_eq!((own.width, own.height), (100, 60));
        // Its own pixels, on white where they were transparent, each
        // ink or paper; enlarged, the edges between them are ramps.
        assert!((own.data[0] - 0.0).abs() < 1e-6 && (own.data[10] - 1.0).abs() < 1e-6);
        assert!(own.data.iter().all(|&v| !(1e-6..=1.0 - 1e-6).contains(&v)));
        assert!(read.data.iter().any(|&v| (0.3..=0.7).contains(&v)));
        // A photo that needs no enlarging is read as it is.
        let fine = luma(RgbaImage::new(640, 480), &Adjust::default());
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
        // Every loader sizes the photo, and so every loader flattens it.
        let loaded = normalize_size(RgbaImage::from_pixel(300, 300, image::Rgba([0, 0, 0, 0])));
        assert!(loaded.pixels().all(|p| p.0 == [255, 255, 255, 255]));
    }
}
