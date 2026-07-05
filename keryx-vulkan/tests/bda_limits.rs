//! Diagnostic (not a fold-correctness test): confirm the sharded weight-blob path actually fits on
//! the device `Vk::new` picks. The mainnet 8B PoM blob is ~4.6 GiB, which exceeds BOTH the 4 GiB
//! `maxStorageBufferRange` descriptor cap and the 2 GiB `maxMemoryAllocationSize` single-allocation
//! cap on AMD — so it ships as multiple ≤1 GiB device-address buffers. This allocates that many
//! 1 GiB device-address shards at once and checks each is usable and distinct.

use keryx_vulkan::Vk;

#[test]
fn bda_sharded_blob_allocates() {
    let vk = match Vk::new() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    eprintln!("device: {}", vk.device_name());

    const GIB: u64 = 1 << 30;
    // A single >2 GiB allocation must still be rejected cleanly (the reason we shard at all).
    match vk.create_device_address_buffer(3 * GIB) {
        Ok((buf, _)) => {
            vk.destroy_buffer(&buf);
            panic!("3 GiB single allocation unexpectedly succeeded — sharding assumption is stale");
        }
        Err(e) => eprintln!("3 GiB single alloc correctly rejected: {e}"),
    }

    // Five 1 GiB shards ≈ 5 GiB total — covers the ~4.6 GiB 8B blob with margin.
    let n_shards = 5u64;
    let mut shards = Vec::new();
    for s in 0..n_shards {
        let (buf, addr) = vk
            .create_device_address_buffer(GIB)
            .unwrap_or_else(|e| panic!("shard {s} (1 GiB) failed: {e}"));
        assert_ne!(addr, 0, "shard {s} got a null device address");
        // Sentinel round-trip: prove the allocation is host-writable/readable.
        let tag = 0xDEAD_BEEF_0000_0000u64 | s;
        vk.write_buffer(&buf, &tag.to_le_bytes());
        let mut back = [0u8; 8];
        vk.read_buffer(&buf, &mut back);
        assert_eq!(u64::from_le_bytes(back), tag, "shard {s} sentinel mismatch");
        eprintln!("shard {s}: 1 GiB OK @ {addr:#x}");
        shards.push((buf, addr));
    }
    let addrs: std::collections::HashSet<u64> = shards.iter().map(|(_, a)| *a).collect();
    assert_eq!(addrs.len(), n_shards as usize, "shard device addresses must be distinct");
    eprintln!("all {n_shards} shards ({} GiB total) resident and distinct", n_shards);

    for (buf, _) in &shards {
        vk.destroy_buffer(buf);
    }
}
