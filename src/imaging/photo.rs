//! Getting a photo to a size the pipeline works at.

use std::path::Path;

use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, ImageResult, RgbaImage};

/// Photos are scaled down to this on their longer side: enough for any
/// symbol, and inside every GPU's texture limit.
pub const MAX_PHOTO_SIDE: u32 = 4096;
/// Photos smaller than this on their shorter side are scaled up, pixel for
/// pixel, so that a symbol rendered at a few pixels a module gets modules
/// the finder scan can see. A long thin rMQR render is judged by its
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
/// transparent (see [`flatten_alpha`]), and at a workable size, large
/// photos reduced smoothly, small ones enlarged by a whole factor with no
/// smoothing, so that their edges stay sharp, as far as the longer side
/// allows. Smooth enlargement was tried for photographs and lost more of
/// the corpus than it gained: a cubic rings at every edge, and even a
/// linear ramp moves the corners the edge tracer finds on crumpled paper.
pub fn normalize_size(mut image: RgbaImage) -> RgbaImage {
    // Before any resampling, which would otherwise bleed the hidden
    // colour of transparent pixels into the edges of opaque ones.
    flatten_alpha(&mut image);
    let longest = image.width().max(image.height()).max(1);
    let shortest = image.width().min(image.height()).max(1);
    if longest > MAX_PHOTO_SIDE {
        let scale = MAX_PHOTO_SIDE as f32 / longest as f32;
        let (w, h) = ((image.width() as f32 * scale) as u32, (image.height() as f32 * scale) as u32);
        image::imageops::resize(&image, w.max(1), h.max(1), FilterType::Triangle)
    } else if shortest < MIN_PHOTO_SIDE {
        let factor = MIN_PHOTO_SIDE.div_ceil(shortest).min(MAX_PHOTO_SIDE / longest).max(1);
        if factor == 1 {
            return image;
        }
        image::imageops::resize(&image, image.width() * factor, image.height() * factor, FilterType::Nearest)
    } else {
        image
    }
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
        // Nearest-neighbour: every enlarged pixel keeps its source value.
        assert_eq!(big.get_pixel(0, 0), tiny.get_pixel(0, 0));
        assert_eq!(big.get_pixel(23, 23), tiny.get_pixel(0, 0));
        assert_eq!(big.get_pixel(24, 0), tiny.get_pixel(1, 0));
        let fine = RgbaImage::new(640, 480);
        assert_eq!(normalize_size(fine.clone()).dimensions(), (640, 480));
        let huge = RgbaImage::new(8192, 4096);
        assert_eq!(normalize_size(huge).dimensions(), (4096, 2048));
        // A long thin render is judged by its height, within the limit on
        // its width.
        let strip = RgbaImage::new(587, 86);
        assert_eq!(normalize_size(strip).dimensions(), (587 * 3, 86 * 3));
        let long_strip = RgbaImage::new(3000, 100);
        assert_eq!(normalize_size(long_strip).dimensions(), (3000, 100), "no whole factor fits");
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
