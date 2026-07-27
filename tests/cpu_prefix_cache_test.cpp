#include "lfm/backend/cpu/prefix_cache.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::vector<std::shared_ptr<lfm::CpuKvPagePool>> pools;
    pools.push_back(std::make_shared<lfm::CpuKvPagePool>(
        lfm::CpuKvCacheMode::Fp32, 4, 4));
    pools.push_back(std::make_shared<lfm::CpuKvPagePool>(
        lfm::CpuKvCacheMode::Fp32, 4, 4));

    lfm::CpuPrefixSnapshot request;
    request.position = 3;
    request.attention_pages.resize(2);
    request.attention_token_counts = {3, 3};
    request.convolution_states = {{1.0f, 2.0f}, {3.0f}};
    request.logits = {4.0f, 5.0f};
    request.seen_tokens = {1, 0, 1};
    std::vector<float> key(4), value(4);
    for (size_t layer = 0; layer < pools.size(); ++layer) {
        const auto page = pools[layer]->allocate();
        request.attention_pages[layer].push_back(page);
        for (size_t token = 0; token < 3; ++token) {
            for (size_t i = 0; i < 4; ++i) {
                key[i] = static_cast<float>(100 * layer + 10 * token + i);
                value[i] = -key[i];
            }
            pools[layer]->write(page, token, key.data(), value.data());
        }
    }

    {
        lfm::CpuPrefixCacheManager cache(pools, 8, 1 << 20);
        const std::vector<int32_t> prefix{10, 20, 30};
        const bool inserted = cache.insert(prefix, request);
        LFM_TEST_CHECK(inserted);
        LFM_TEST_CHECK(cache.size() == 1);
        const auto hit = cache.acquire({10, 20, 30, 40});
        LFM_TEST_CHECK(hit);
        LFM_TEST_CHECK(hit->matched_tokens == 3);
        LFM_TEST_CHECK(hit->snapshot.attention_pages[0].back() !=
               request.attention_pages[0].back());
        LFM_TEST_CHECK(pools[0]->key_fp32(hit->snapshot.attention_pages[0].back(), 2)[1] == 21.0f);

        // Mutating the private request tail must not modify the cached page.
        std::fill(key.begin(), key.end(), 999.0f);
        std::fill(value.begin(), value.end(), -999.0f);
        pools[0]->write(hit->snapshot.attention_pages[0].back(), 3,
                        key.data(), value.data());
        const auto second = cache.acquire(prefix);
        LFM_TEST_CHECK(second);
        LFM_TEST_CHECK(pools[0]->key_fp32(second->snapshot.attention_pages[0].back(), 2)[1] == 21.0f);
        LFM_TEST_CHECK(cache.metrics().partial_hits == 1);
        LFM_TEST_CHECK(cache.metrics().cow_pages >= 4);

        for (size_t layer = 0; layer < pools.size(); ++layer) {
            for (auto page : hit->snapshot.attention_pages[layer]) pools[layer]->release(page);
            for (auto page : second->snapshot.attention_pages[layer]) pools[layer]->release(page);
        }
    }
    for (size_t layer = 0; layer < pools.size(); ++layer) {
        for (auto page : request.attention_pages[layer]) pools[layer]->release(page);
        LFM_TEST_CHECK(pools[layer]->stats().used_pages == 0);
    }
    std::cout << "cpu_prefix_cache_test: ok\n";
}
