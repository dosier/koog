package ai.koog.prompt.message

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.time.Clock
import kotlin.time.Instant

class ResponseMetaInfoTest {

    private val fixedClock = object : Clock {
        override fun now(): Instant = Instant.fromEpochMilliseconds(1700000000000L)
    }

    @Test
    fun testCreateWithAllFields() {
        val metaInfo = ResponseMetaInfo.create(
            clock = fixedClock,
            totalTokensCount = 150,
            inputTokensCount = 100,
            outputTokensCount = 50,
            cacheCreationTokens = 80,
            cacheReadTokens = 20
        )

        assertEquals(150, metaInfo.totalTokensCount)
        assertEquals(100, metaInfo.inputTokensCount)
        assertEquals(50, metaInfo.outputTokensCount)
        assertEquals(80, metaInfo.cacheCreationTokens)
        assertEquals(20, metaInfo.cacheReadTokens)
    }

    @Test
    fun testCreateWithCacheMetricsOnly() {
        val metaInfo = ResponseMetaInfo.create(
            clock = fixedClock,
            cacheCreationTokens = 500,
            cacheReadTokens = 300
        )

        assertNull(metaInfo.totalTokensCount)
        assertNull(metaInfo.inputTokensCount)
        assertNull(metaInfo.outputTokensCount)
        assertEquals(500, metaInfo.cacheCreationTokens)
        assertEquals(300, metaInfo.cacheReadTokens)
    }

    @Test
    fun testCreateWithoutCacheMetrics() {
        val metaInfo = ResponseMetaInfo.create(
            clock = fixedClock,
            totalTokensCount = 100,
            inputTokensCount = 60,
            outputTokensCount = 40
        )

        assertEquals(100, metaInfo.totalTokensCount)
        assertEquals(60, metaInfo.inputTokensCount)
        assertEquals(40, metaInfo.outputTokensCount)
        assertNull(metaInfo.cacheCreationTokens)
        assertNull(metaInfo.cacheReadTokens)
    }

    @Test
    fun testEmptyResponseMetaInfo() {
        val empty = ResponseMetaInfo.Empty

        assertNull(empty.totalTokensCount)
        assertNull(empty.inputTokensCount)
        assertNull(empty.outputTokensCount)
        assertNull(empty.cacheCreationTokens)
        assertNull(empty.cacheReadTokens)
    }

    @Test
    fun testCacheHitScenario() {
        // Simulates a cache hit where tokens are read from cache
        val metaInfo = ResponseMetaInfo.create(
            clock = fixedClock,
            totalTokensCount = 150,
            inputTokensCount = 100,
            outputTokensCount = 50,
            cacheCreationTokens = 0,  // No new tokens written to cache
            cacheReadTokens = 80      // 80 tokens read from cache
        )

        assertEquals(0, metaInfo.cacheCreationTokens)
        assertEquals(80, metaInfo.cacheReadTokens)
    }

    @Test
    fun testCacheMissScenario() {
        // Simulates a cache miss where tokens are written to cache
        val metaInfo = ResponseMetaInfo.create(
            clock = fixedClock,
            totalTokensCount = 150,
            inputTokensCount = 100,
            outputTokensCount = 50,
            cacheCreationTokens = 80,  // 80 tokens written to cache
            cacheReadTokens = 0        // No tokens read from cache
        )

        assertEquals(80, metaInfo.cacheCreationTokens)
        assertEquals(0, metaInfo.cacheReadTokens)
    }

    @Test
    fun testPartialCacheHitScenario() {
        // Simulates a partial cache hit where some tokens are read and some are written
        val metaInfo = ResponseMetaInfo.create(
            clock = fixedClock,
            totalTokensCount = 200,
            inputTokensCount = 150,
            outputTokensCount = 50,
            cacheCreationTokens = 50,  // 50 new tokens written to cache
            cacheReadTokens = 100      // 100 tokens read from cache
        )

        assertEquals(50, metaInfo.cacheCreationTokens)
        assertEquals(100, metaInfo.cacheReadTokens)
    }

    @Test
    fun testDataClassEquality() {
        val metaInfo1 = ResponseMetaInfo(
            timestamp = fixedClock.now(),
            totalTokensCount = 100,
            inputTokensCount = 60,
            outputTokensCount = 40,
            cacheCreationTokens = 30,
            cacheReadTokens = 30
        )

        val metaInfo2 = ResponseMetaInfo(
            timestamp = fixedClock.now(),
            totalTokensCount = 100,
            inputTokensCount = 60,
            outputTokensCount = 40,
            cacheCreationTokens = 30,
            cacheReadTokens = 30
        )

        assertEquals(metaInfo1, metaInfo2)
    }

    @Test
    fun testDataClassCopy() {
        val original = ResponseMetaInfo(
            timestamp = fixedClock.now(),
            totalTokensCount = 100,
            inputTokensCount = 60,
            outputTokensCount = 40,
            cacheCreationTokens = null,
            cacheReadTokens = null
        )

        val updated = original.copy(
            cacheCreationTokens = 50,
            cacheReadTokens = 10
        )

        assertEquals(100, updated.totalTokensCount)
        assertEquals(60, updated.inputTokensCount)
        assertEquals(40, updated.outputTokensCount)
        assertEquals(50, updated.cacheCreationTokens)
        assertEquals(10, updated.cacheReadTokens)
    }
}
