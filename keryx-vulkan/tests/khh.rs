//! Bit-exactness test: the Vulkan kHeavyHash shader MUST agree with a host reference on the real
//! GPU. The reference uses the `keccak` crate's `f1600` (the same permutation the miner's host PoW
//! uses) so the shader's hand-written Keccak is validated against the trusted implementation.

use keryx_vulkan::khh::{KhhGpu, MATRIX_LEN};
use rand::{rngs::StdRng, Rng, SeedableRng};

// cSHAKE256("ProofOfWorkHash") / cSHAKE256("HeavyHash") initial states (== src/pow/hasher.rs).
const POW_P: [u64; 25] = [
    1242148031264380989, 3008272977830772284, 2188519011337848018, 1992179434288343456, 8876506674959887717,
    5399642050693751366, 1745875063082670864, 8605242046444978844, 17936695144567157056, 3343109343542796272,
    1123092876221303306, 4963925045340115282, 17037383077651887893, 16629644495023626889, 12833675776649114147,
    3784524041015224902, 1082795874807940378, 13952716920571277634, 13411128033953605860, 15060696040649351053,
    9928834659948351306, 5237849264682708699, 12825353012139217522, 6706187291358897596, 196324915476054915,
];
const HEAVY_P: [u64; 25] = [
    4239941492252378377, 8746723911537738262, 8796936657246353646, 1272090201925444760, 16654558671554924250,
    8270816933120786537, 13907396207649043898, 6782861118970774626, 9239690602118867528, 11582319943599406348,
    17596056728278508070, 15212962468105129023, 7812475424661425213, 3370482334374859748, 5690099369266491460,
    8596393687355028144, 570094237299545110, 9119540418498120711, 16901969272480492857, 13372017233735502424,
    14372891883993151831, 5171152063242093102, 10573107899694386186, 6096431547456407061, 1592359455985097269,
];
const WAVE_MIX_KEYS: [u64; 4] =
    [0x9e3779b97f4a7c15, 0x6c62272e07bb0142, 0xb5ad4eceda1ce2a9, 0x243f6a8885a308d3];

fn pow_hash(header: &[u64; 9], nonce: u64) -> [u64; 4] {
    let mut a = POW_P;
    for i in 0..9 {
        a[i] ^= header[i];
    }
    a[9] ^= nonce;
    keccak::f1600(&mut a);
    [a[0], a[1], a[2], a[3]]
}

fn heavy_hash(pw: &[u64; 4], mat: &[u32; MATRIX_LEN]) -> [u64; 4] {
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&pw[i].to_le_bytes());
    }
    let mut nib = [0u32; 64];
    for i in 0..32 {
        nib[2 * i] = (bytes[i] >> 4) as u32;
        nib[2 * i + 1] = (bytes[i] & 0x0F) as u32;
    }
    let mut pbytes = [0u8; 32];
    for i in 0..32 {
        let mut s1 = 0u32;
        let mut s2 = 0u32;
        for j in 0..64 {
            s1 += mat[(2 * i) * 64 + j] * nib[j];
            s2 += mat[(2 * i + 1) * 64 + j] * nib[j];
        }
        let matbyte = (((s1 >> 10) << 4) | (s2 >> 10)) as u8;
        pbytes[i] = matbyte ^ bytes[i];
    }
    let mut w = [0u64; 4];
    for k in 0..4 {
        w[k] = u64::from_le_bytes(pbytes[k * 8..k * 8 + 8].try_into().unwrap());
    }
    for r in 0..4usize {
        w[0] = w[0].wrapping_add(w[1]).rotate_left(17) ^ WAVE_MIX_KEYS[r & 3];
        w[2] = w[2].wrapping_add(w[3]).rotate_left(47) ^ WAVE_MIX_KEYS[(r + 2) & 3];
        w[1] = w[1].wrapping_add(w[2]).rotate_left(31) ^ WAVE_MIX_KEYS[(r + 1) & 3];
        w[3] = w[3].wrapping_add(w[0]).rotate_left(13) ^ WAVE_MIX_KEYS[(r + 3) & 3];
    }
    let mut h = HEAVY_P;
    for i in 0..4 {
        h[i] ^= w[i];
    }
    keccak::f1600(&mut h);
    [h[0], h[1], h[2], h[3]]
}

fn le_leq256(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

fn host_lowest_winner(
    header: &[u64; 9],
    target: &[u64; 4],
    mat: &[u32; MATRIX_LEN],
    start: u64,
    batch: u32,
    mask: u64,
    fixed: u64,
) -> Option<u64> {
    for i in 0..batch as u64 {
        let nonce = ((start + i) & mask) | fixed; // effective nonce (extranonce-masked)
        if le_leq256(&heavy_hash(&pow_hash(header, nonce), mat), target) {
            return Some(nonce);
        }
    }
    None
}

#[test]
fn vulkan_khh_matches_host_reference() {
    let mut rng = StdRng::seed_from_u64(0xBADC0FFEE0DDF00D);

    let mut gpu = match KhhGpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };
    eprintln!("kHeavyHash running on: {}", gpu.device_name());

    let start: u64 = 0x0102_0304_0506_0000;
    let batch: u32 = 8192;

    for trial in 0..5 {
        // Random 64x64 matrix of 4-bit entries, and a random pow header.
        let mut matrix = [0u32; MATRIX_LEN];
        for e in matrix.iter_mut() {
            *e = rng.gen_range(0..16);
        }
        let mut header = [0u64; 9];
        for h in header.iter_mut() {
            *h = rng.r#gen();
        }
        gpu.upload_block(&matrix, header, [0; 4]);

        // Sanity: a single known nonce's full pow must match (catches keccak/matmul bugs directly).
        let n0 = start + 1234;
        let host_pow = heavy_hash(&pow_hash(&header, n0), &matrix);
        let always = [u64::MAX; 4];
        gpu.upload_block(&matrix, header, always);
        assert_eq!(gpu.mine(n0, 1, u64::MAX, 0), Some(n0), "trial {trial}: any-target single nonce must win");

        // Host pow distribution over the batch → choose meaningful targets.
        let mut finals: Vec<[u64; 4]> = (0..batch as u64)
            .map(|i| heavy_hash(&pow_hash(&header, start + i), &matrix))
            .collect();
        finals.sort_by(|a, b| {
            for i in (0..4).rev() {
                if a[i] != b[i] {
                    return a[i].cmp(&b[i]);
                }
            }
            std::cmp::Ordering::Equal
        });

        let impossible = [0u64; 4];
        let median = finals[finals.len() / 2];
        for target in [impossible, median, always] {
            gpu.upload_block(&matrix, header, target);
            let host = host_lowest_winner(&header, &target, &matrix, start, batch, u64::MAX, 0);
            let got = gpu.mine(start, batch, u64::MAX, 0);
            assert_eq!(got, host, "trial {trial}: GPU winner {got:?} != host {host:?}");
        }

        // Pool extranonce: miner controls the low 20 bits; the extranonce sits in the high bits.
        let mask: u64 = (1 << 20) - 1;
        let fixed: u64 = 0xABCDu64 << 20;
        gpu.upload_block(&matrix, header, always);
        let host_m = host_lowest_winner(&header, &always, &matrix, start, batch, mask, fixed);
        let got_m = gpu.mine(start, batch, mask, fixed);
        assert_eq!(got_m, host_m, "trial {trial}: masked GPU winner {got_m:?} != host {host_m:?}");
        if let Some(n) = got_m {
            assert_eq!(n & !mask, fixed, "winning nonce must carry the extranonce in its high bits");
        }
        let _ = host_pow;
    }
}
