//! The window's textures of the photo: laid on the graphics card the
//! photo is adjusted on, where the window is drawn by it
//! ([`qr_forensics::imaging::gpu::shared_device`]), and egui's own
//! otherwise. What the adjustments make there to be shown is copied into
//! the texture where it lies ([`OnCard::copy_to`]), and what the cores
//! made, the photo in grey, is handed the texture as it is, without
//! egui's colours being made of it first.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use qr_forensics::imaging::gpu::{OnCard, Shown};

/// The window's drawing, where the graphics card the photo is adjusted
/// on draws it.
#[derive(Clone)]
pub struct Screen {
    render: egui_wgpu::RenderState,
    /// For a texture of egui's own, for a photo too large to be drawn
    /// through one of the card's whole.
    ctx: egui::Context,
    dropped: Dropped,
}

/// The textures of the card's dropped since the window was last drawn,
/// which the frame being laid out may have drawn through: let go of once it
/// is drawn ([`Screen::let_go`]), as egui lets go of its own. Let go of as
/// they were dropped, a photo enlarged or undone, or its binarised view
/// made anew, by what the end of a frame applied was missing from that
/// frame, which drew through the texture it had replaced.
type Dropped = Arc<egui::mutex::Mutex<Vec<egui::TextureId>>>;

/// A texture of the window's on the card, registered with egui to be drawn
/// through, and let go of once it is dropped and the frame that may draw
/// through it is drawn.
pub struct CardTexture {
    id: egui::TextureId,
    texture: wgpu::Texture,
    size: [usize; 2],
    dropped: Dropped,
}

impl Drop for CardTexture {
    fn drop(&mut self) {
        self.dropped.lock().push(self.id);
    }
}

/// A texture of the window's: egui's own, or one on the card.
pub enum Texture {
    Egui(egui::TextureHandle),
    Card(CardTexture),
}

impl Texture {
    pub fn id(&self) -> egui::TextureId {
        match self {
            Texture::Egui(handle) => handle.id(),
            Texture::Card(texture) => texture.id,
        }
    }

    /// Its size, in its own pixels: what the tests check a texture by.
    #[cfg(test)]
    pub fn size(&self) -> [usize; 2] {
        match self {
            Texture::Egui(handle) => handle.size(),
            Texture::Card(texture) => texture.size,
        }
    }
}

/// What a texture is to show, four bytes to each pixel: left on the card,
/// or held by the cores.
pub enum Pixels<'a> {
    Card(&'a OnCard),
    Bytes(&'a [u8]),
}

impl<'a> From<&'a Shown> for Pixels<'a> {
    fn from(shown: &'a Shown) -> Self {
        match shown {
            Shown::Card(on_card) => Pixels::Card(on_card),
            Shown::Bytes(bytes) => Pixels::Bytes(bytes),
        }
    }
}

impl Screen {
    /// The window's drawing, where it is the card's own: what the program
    /// asks for as it starts ([`qr_forensics::imaging::gpu::shared_device`]),
    /// and `None` where the window was drawn otherwise after all.
    pub fn of(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        let render = cc.wgpu_render_state.clone()?;
        qr_forensics::imaging::gpu::is_card(&render.device).then(|| Self { render, ctx: cc.egui_ctx.clone(), dropped: Dropped::default() })
    }

    /// The textures of the card's dropped since this was last asked let go
    /// of: what each frame asks first, once the last has been drawn
    /// ([`Dropped`]).
    pub fn let_go(&self) {
        let dropped = std::mem::take(&mut *self.dropped.lock());
        if !dropped.is_empty() {
            let mut renderer = self.render.renderer.write();
            for id in &dropped {
                renderer.free_texture(id);
            }
        }
    }

    /// `pixels`, `width` by `height`, as a new texture of the card's.
    pub fn laid(&self, pixels: Pixels<'_>, (width, height): (usize, usize)) -> CardTexture {
        let texture = self.texture((width, height));
        self.copied(&texture, pixels, (width, height));
        texture
    }

    /// `pixels`, `width` by `height`, laid into `texture` where it is one
    /// of the card's of that size, and into a new one of the card's in its
    /// place otherwise.
    pub fn lay(&self, texture: &mut Texture, pixels: Pixels<'_>, (width, height): (usize, usize)) {
        match texture {
            Texture::Card(card) if card.size == [width, height] => self.copied(card, pixels, (width, height)),
            _ => *texture = Texture::Card(self.laid(pixels, (width, height))),
        }
    }

    /// `pixels` copied into `texture`, of their size: on the card from
    /// where they lie there, or handed it from the cores.
    fn copied(&self, texture: &CardTexture, pixels: Pixels<'_>, (width, height): (usize, usize)) {
        match pixels {
            Pixels::Card(on_card) => {
                assert_eq!((on_card.width, on_card.height), (width, height));
                on_card.copy_to(&texture.texture);
            }
            Pixels::Bytes(bytes) => {
                assert_eq!(bytes.len(), width * height * 4);
                self.render.queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &texture.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    bytes,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((width * 4) as u32), rows_per_image: Some(height as u32) },
                    wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
                );
            }
        }
    }

    /// A new texture of the card's `width` by `height`, of the format egui
    /// draws its own through, registered with it.
    fn texture(&self, (width, height): (usize, usize)) -> CardTexture {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        // Copied from as well, where the tests read it back ([`Self::read`]).
        #[cfg(test)]
        let usage = usage | wgpu::TextureUsages::COPY_SRC;
        let texture = self.render.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("QR Forensics photo"),
            size: wgpu::Extent3d { width: width.max(1) as u32, height: height.max(1) as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = self.render.renderer.write().register_native_texture(&self.render.device, &view, wgpu::FilterMode::Linear);
        CardTexture { id, texture, size: [width, height], dropped: self.dropped.clone() }
    }

    /// The window's context, for a texture of egui's own.
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    /// The drawing of a window on the card the photo is adjusted on, as
    /// the program asks for it as it starts, with no window to draw: what
    /// the tests lay textures through.
    #[cfg(test)]
    pub fn on_card(ctx: &egui::Context) -> Self {
        let (instance, adapter, device, queue) = qr_forensics::imaging::gpu::shared_device().expect("a card");
        let config = egui_wgpu::WgpuConfiguration { wgpu_setup: egui_wgpu::WgpuSetupExisting { instance: instance.clone(), adapter, device, queue }.into(), ..Default::default() };
        let render = cubecl::future::block_on(egui_wgpu::RenderState::create(&config, &instance, None, egui_wgpu::RendererOptions::default())).expect("drawn by the card");
        assert!(qr_forensics::imaging::gpu::is_card(&render.device));
        Self { render, ctx: ctx.clone(), dropped: Dropped::default() }
    }

    /// What `texture`, one of the card's, shows, four bytes to each pixel,
    /// row after row: what the tests check one by.
    #[cfg(test)]
    pub fn read(&self, texture: &Texture) -> Vec<u8> {
        let Texture::Card(texture) = texture else { panic!("a texture of the card's") };
        let [width, height] = texture.size;
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let (device, queue) = (&self.render.device, &self.render.queue);
        let back = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * height) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &texture.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &back, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row as u32), rows_per_image: Some(height as u32) } },
            wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        back.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("mapped"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("the copy");
        let mapped = back.slice(..).get_mapped_range().expect("mapped");
        mapped.chunks(row).flat_map(|laid| laid[..width * 4].to_vec()).collect()
    }

    /// Whether egui draws `id` through a texture of the card's: one laid
    /// here and not yet let go of.
    #[cfg(test)]
    pub fn registered(&self, id: egui::TextureId) -> bool {
        self.render.renderer.read().texture(&id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qr_forensics::imaging::SharedRgba;
    use qr_forensics::imaging::adjust::Adjust;

    /// Opaque pixels `width` by `height`, each its own.
    fn pixels(width: usize, height: usize) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let word = (i as u32).wrapping_mul(2_654_435_761);
                [word as u8, (word >> 8) as u8, (word >> 16) as u8, 255]
            })
            .collect()
    }

    #[test]
    fn a_texture_on_the_card_shows_what_is_laid_into_it_and_is_let_go_of() {
        let ctx = egui::Context::default();
        let screen = Screen::on_card(&ctx);
        // Rows of a width that a copy to a texture lays out longer.
        let (width, height) = (70, 9);
        let first = pixels(width, height);
        let mut texture = Texture::Card(screen.laid(Pixels::Bytes(&first), (width, height)));
        let id = texture.id();
        assert!(screen.registered(id));
        assert_eq!((texture.size(), screen.read(&texture)), ([width, height], first.clone()));
        // Laid again at the same size, from where an adjustment left them on
        // the card: into the same texture, as the adjustment made them.
        let rgba: Vec<u8> = first.iter().map(|&byte| byte / 2 + 40).collect();
        let shared: SharedRgba = Arc::new(rgba.clone());
        let (shown, _) = Adjust::default().apply_to_shared_shown(&shared, width, height, [0, 0, width, height]);
        assert!(matches!(shown, Shown::Card(_)), "left on the card");
        screen.lay(&mut texture, (&shown).into(), (width, height));
        assert_eq!(texture.id(), id);
        assert_eq!(screen.read(&texture), Adjust::default().apply_to(&rgba, width, height, [0, 0, width, height]).0);
        // At another size, into a texture of its own in its place, and the
        // old one let go of once the frame that may draw through it is.
        let small = pixels(5, 4);
        screen.lay(&mut texture, Pixels::Bytes(&small), (5, 4));
        assert!(texture.id() != id && screen.registered(id));
        screen.let_go();
        assert!(!screen.registered(id));
        assert_eq!((texture.size(), screen.read(&texture)), ([5, 4], small));
        // One of egui's own, laid into, is one of the card's after.
        let mut own = Texture::Egui(ctx.load_texture("own", egui::ColorImage::filled([width, height], egui::Color32::WHITE), egui::TextureOptions::LINEAR));
        screen.lay(&mut own, Pixels::Bytes(&first), (width, height));
        assert!(matches!(own, Texture::Card(_)));
        assert_eq!(screen.read(&own), first);
        // Each let go of once it is dropped and the frame drawn.
        let ids = [texture.id(), own.id()];
        drop((texture, own));
        assert!(ids.iter().all(|&id| screen.registered(id)));
        screen.let_go();
        assert!(!ids.iter().any(|&id| screen.registered(id)));
    }
}
