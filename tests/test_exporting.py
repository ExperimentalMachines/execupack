from pipeline import exporting


def meminfo(path, swap_free_kb, available_kb):
    path.write_text(
        f"MemTotal:       16373452 kB\nMemAvailable:   {available_kb} kB\n"
        f"SwapTotal:      25165820 kB\nSwapFree:       {swap_free_kb} kB\n"
    )


def test_memory_sampler_keeps_the_peak_swap_and_the_lowest_available_memory(tmp_path):
    path = tmp_path / "meminfo"
    meminfo(path, swap_free_kb=25165820, available_kb=12000000)
    sampler = exporting.MemorySampler(interval=3600, meminfo=path)
    sampler.sample()
    meminfo(path, swap_free_kb=20165820, available_kb=300000)  # 5,000,000 kB of swap in use
    sampler.sample()
    meminfo(path, swap_free_kb=25165820, available_kb=9000000)
    sampler.sample()
    result = sampler.result()
    assert result["peak_swap_used_bytes"] == 5_000_000 * 1024
    assert result["min_mem_available_bytes"] == 300_000 * 1024


def test_memory_sampler_keeps_the_peak_of_ram_plus_swap_within_one_sample(tmp_path):
    path = tmp_path / "meminfo"
    sampler = exporting.MemorySampler(interval=3600, meminfo=path)
    for swap_used_kb, available_kb in ((5_000_000, 9_000_000), (1_000_000, 300_000), (4_000_000, 1_000_000)):
        meminfo(path, swap_free_kb=25165820 - swap_used_kb, available_kb=available_kb)
        sampler.sample()
    result = sampler.result()
    # 16,373,452 - 1,000,000 + 4,000,000 kB, the third sample. The separate extremes would
    # add up to 16,373,452 - 300,000 + 5,000,000 = 21,073,452 kB, which never happened.
    assert result["peak_in_use_bytes"] == 19_373_452 * 1024
    assert result["peak_in_use_at"].endswith("Z")


def test_memory_sampler_is_a_no_op_without_proc(tmp_path):
    with exporting.MemorySampler(interval=3600, meminfo=tmp_path / "missing", cgroup=()) as sampler:
        pass
    assert set(sampler.result().values()) == {None}


def test_contains_is_shared_by_the_npu_backends(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"x" * 100 + b"NeuropilotBackend" + b"y" * 100)
    assert exporting.contains(path, b"NeuropilotBackend", block=7)
