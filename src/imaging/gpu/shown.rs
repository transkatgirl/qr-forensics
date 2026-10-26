//! What the adjustments make of a photo to be shown, left on the
//! graphics card for the window to lay into its own texture there
//! ([`super::shared_device`]).
//!
//! The window was handed the photo as it is shown by taking it off the
//! card, four bytes to each pixel, laying the bytes out again as egui's
//! colours, and handing them to the card once more as a texture: for a
//! photo of twelve megapixels, some fifteen milliseconds each time a
//! slider moved, nearly all of it the bytes' way through the cores'
//! memory and back. Drawn with the card the photo is adjusted on, the
//! window has the pixels copied from where they lie into its texture,
//! by the card itself.

use super::*;

/// Words a row of an image of `width` pixels, four bytes to each, takes
/// in a buffer copied to a texture: as many as come to a whole number of
/// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] bytes. What is made to be
/// shown is laid out so where it can be ([`OnCard::padded`]), and is then
/// copied to a texture as it lies.
pub(super) fn stride(width: usize) -> usize {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize) / 4
}

/// `count` pixels of `words`, rows `width` across, laid out `stride` words
/// to a row.
#[cube(launch)]
fn padded_kernel(words: &[u32], padded: &mut [u32], width: u32, stride: u32, count: u32) {
    let i = line_place();
    if i < count {
        padded[((i / width) * stride + i % width) as usize] = words[i as usize];
    }
}

/// `count` values of `luma`, rows `width` across, as opaque grey pixels,
/// four bytes to a word, laid out `across` words to a row ([`grey_word`]).
#[cube(launch)]
fn greyed_kernel(luma: &[f32], words: &mut [u32], width: u32, across: u32, count: u32) {
    let i = line_place();
    if i < count {
        words[((i / width) * across + i % width) as usize] = grey_word(luma[i as usize]);
    }
}

/// Pixels to be shown, `width` by `height`, four bytes to each in a word,
/// on the card.
pub struct OnCard {
    words: Held,
    pub width: usize,
    pub height: usize,
    /// Words from the start of a row to the next: the width, or as many as
    /// a copy to a texture wants a row laid out in ([`stride`]).
    across: usize,
}

impl OnCard {
    /// Pixels made row after row, which are laid out again to be copied to
    /// a texture: what the tests copy them so from.
    #[cfg(test)]
    pub(super) fn of(words: Held, (width, height): (usize, usize)) -> Self {
        Self { words, width, height, across: width }
    }

    /// Pixels made a row to every [`stride`] words, as a copy to a texture
    /// wants them, which are copied to one as they lie.
    pub(super) fn padded(words: Held, (width, height): (usize, usize)) -> Self {
        Self { words, width, height, across: stride(width) }
    }

    /// The opaque grey of `luma`, an image of a channel on the card.
    pub(crate) fn greyed(luma: &Image) -> Self {
        let client = client();
        let count = luma.width * luma.height;
        let across = stride(luma.width);
        let words = Held::empty(client, across * luma.height);
        greyed_kernel::launch(client, line_cubes(count), CubeDim::new_1d(LINE), luma.values.arg(), words.arg(), luma.width as u32, across as u32, count as u32);
        Self::padded(words, (luma.width, luma.height))
    }

    /// The pixels, off the card, row after row.
    pub fn read(&self) -> Vec<u8> {
        let row = self.width * 4;
        if self.across == self.width {
            return self.words.read_bytes(client(), row * self.height);
        }
        let laid = self.words.read_bytes(client(), self.across * 4 * self.height);
        let mut out = Vec::with_capacity(row * self.height);
        for laid in laid.chunks_exact(self.across * 4) {
            out.extend_from_slice(&laid[..row]);
        }
        out
    }

    /// The pixels laid into `texture`, an `Rgba8Unorm` texture of the card
    /// ([`super::shared_device`]) at least as large as they are, which they
    /// may be copied to: copied there on the card, after the work that made
    /// them, on whichever thread it was asked for, rows laid out first as a
    /// copy to a texture wants them where they were not made so.
    pub fn copy_to(&self, texture: &wgpu::Texture) {
        let card = card().as_ref().unwrap_or_else(|why| panic!("{why}"));
        let client = &card.client;
        let (width, height) = (self.width, self.height);
        if width == 0 || height == 0 {
            return;
        }
        let across = stride(width);
        let laid = if self.across == across {
            self.words.clone()
        } else {
            let laid = Held::empty(client, across * height);
            padded_kernel::launch(client, line_cubes(width * height), CubeDim::new_1d(LINE), self.words.arg(), laid.arg(), width as u32, across as u32, (width * height) as u32);
            laid
        };
        // Where the buffer lies, held in place while it is copied from.
        // Asked for, it hands the queue the work held back on this thread's
        // stream and on the stream the buffer was made on, whichever thread
        // made it, and the flush hands it this thread's once more: the copy,
        // handed the queue after, follows the work that makes the pixels
        // wherever they were made
        // ([`tests::pixels_made_on_another_thread_are_copied_as_made`]).
        let resource = off_card(client.get_resource::<WgpuServer<AutoCompiler>>(laid.handle.clone()));
        off_card(client.flush());
        let buffer = resource.resource();
        let mut encoder = card.setup.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("QR Forensics shown") });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer.buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: buffer.offset, bytes_per_row: Some((across * 4) as u32), rows_per_image: Some(height as u32) },
            },
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        );
        card.setup.queue.submit([encoder.finish()]);
    }
}

/// Pixels to be shown, four bytes to each, as an adjustment made them:
/// left on the card where it made them whole there, or taken off it.
pub enum Shown {
    Bytes(Vec<u8>),
    Card(OnCard),
}

impl Shown {
    /// The pixels, off the card where they are on it.
    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            Shown::Bytes(bytes) => bytes,
            Shown::Card(on_card) => on_card.read(),
        }
    }
}

impl Toned {
    /// What is shown of the part, left on the card, and what is read of it,
    /// off the card: as [`Toned::taken`] takes them, the part as it was
    /// toned to be shown, in its colours or its grey ([`Tone`]); or, where
    /// the luminance was `filtered`, that, and its grey made on the card.
    pub(crate) fn kept(self, filtered: Option<Image>) -> (OnCard, Vec<f32>) {
        let (width, height) = (self.luma.width, self.luma.height);
        match filtered {
            None => {
                assert_ne!(self.tone, Tone::Nothing, "a part kept to be shown is toned to be shown");
                let luma = self.luma.read();
                (OnCard { words: self.shown, width, height, across: self.tone.across(width) }, luma)
            }
            Some(filtered) => (OnCard::greyed(&filtered), filtered.read()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_kept_on_the_card_is_what_is_taken_off_it() {
        let (width, height) = (37, 23);
        let rgba: Vec<u8> = (0..width * height * 4).map(|i| (i * 37 % 251) as u8).collect();
        let source = Source { rgba: &rgba, width, part: [0, 0, width, height], shared: None };
        let table: [u8; 256] = std::array::from_fn(|i| 255 - i as u8);
        let toned = |tone: Tone| adjusted_within(usize::MAX, &source, &[], &table, None, tone).expect("fits");
        let (shown, luma) = toned(Tone::Colours { padded: false }).taken(None, true);
        // In its colours laid out as a copy to a texture wants them, which
        // for a part this wide is not row after row, and row after row:
        // either way, what is taken.
        assert_ne!(stride(width), width);
        for padded in [true, false] {
            let (kept, kept_luma) = toned(Tone::Colours { padded }).kept(None);
            assert!(kept.read() == shown && kept_luma == luma, "padded: {padded}");
        }
        // In grey, the grey of the luminance, as the cores grey it.
        let (kept, kept_luma) = toned(Tone::Grey).kept(None);
        let mut grey = vec![0u8; width * height * 4];
        crate::imaging::greyed(&luma, &mut grey);
        assert!(kept.read() == grey && kept_luma == luma, "the grey of the luminance as it is toned is the cores' grey of it");
        // Toned for nothing to be shown, the same luminance alone.
        assert!(toned(Tone::Nothing).taken(None, false) == (Vec::new(), luma.clone()));
        // Filtered, nothing is toned to be shown, and the filter's grey is.
        let filtered = toned(Tone::Nothing).luma.median(1);
        let (shown, luma) = toned(Tone::Nothing).taken(Some(filtered.clone()), true);
        let (kept, kept_luma) = toned(Tone::Nothing).kept(Some(filtered));
        assert!(kept.read() == shown && kept_luma == luma, "the grey of a filter on the card is the cores' grey of it");
        assert_eq!(Shown::Card(kept).into_bytes(), shown);
    }

    /// `on_card` copied into a texture of the card its size, and read back
    /// out of it, a word to a pixel, row after row.
    fn copied(on_card: &OnCard) -> Vec<u32> {
        let (_, _, device, queue) = shared_device().expect("a card");
        let (width, height) = (on_card.width, on_card.height);
        let across = stride(width);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        on_card.copy_to(&texture);
        let back = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (across * 4 * height) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &back, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((across * 4) as u32), rows_per_image: Some(height as u32) } },
            wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        back.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("mapped"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("the copy");
        let mapped = back.slice(..).get_mapped_range().expect("mapped");
        mapped.as_chunks::<4>().0.chunks(across).flat_map(|row| row[..width].iter().map(|&b| u32::from_le_bytes(b))).collect()
    }

    #[test]
    fn rows_are_laid_out_as_a_copy_to_a_texture_wants_them() {
        assert_eq!(stride(64), 64);
        assert_eq!(stride(1), 64);
        assert_eq!(stride(65), 128);
        // Copied into a texture of the card and read back out of it.
        let (width, height) = (70, 9);
        let across = stride(width);
        let words: Vec<u32> = (0..(width * height) as u32).map(|i| i.wrapping_mul(2_654_435_761) | 0xff00_0000).collect();
        assert_eq!(copied(&OnCard::of(Held::of(client(), &words), (width, height))), words);
        // Made laid out so, with whatever lies past each row, they are
        // copied as they lie, and read off the card a row at a time.
        let laid: Vec<u32> = (0..across * height).map(|i| if i % across < width { words[i / across * width + i % across] } else { 0xdead_beef }).collect();
        let padded = OnCard::padded(Held::of(client(), &laid), (width, height));
        assert_eq!(copied(&padded), words);
        assert_eq!(padded.read(), words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<u8>>());
    }

    #[test]
    fn pixels_made_on_another_thread_are_copied_as_made() {
        // The grey of a luminance made by the work of a thread of its own,
        // held back on that thread's stream until something hands it the
        // queue, and copied on this one.
        let (width, height) = (70, 9);
        let luma: Vec<f32> = (0..width * height).map(|i| (i * 37 % 251) as f32 / 250.0).collect();
        let made = {
            let luma = luma.clone();
            std::thread::spawn(move || OnCard::greyed(&Image::handed(&luma, width, height))).join().expect("the grey made")
        };
        let mut grey = vec![0u8; width * height * 4];
        crate::imaging::greyed(&luma, &mut grey);
        assert_eq!(copied(&made), grey.as_chunks::<4>().0.iter().map(|&bytes| u32::from_le_bytes(bytes)).collect::<Vec<u32>>());
    }
}
