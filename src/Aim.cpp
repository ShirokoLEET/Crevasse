#include "Aim.h"
#include <Windows.h>
#include <obs-module.h>
#include <cmath>
#include <algorithm>

Aim::Aim()
{
	// 获取屏幕分辨率
	m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
	m_screenHeight = GetSystemMetrics(SM_CYSCREEN);
	blog(LOG_INFO, "[Aim] Screen size: %dx%d", m_screenWidth, m_screenHeight);

	resetStickyState();
}

Aim::~Aim()
{
	stop();
}

void Aim::start()
{
	if (m_running.load()) {
		return; // 已经在运行
	}

	m_shouldStop.store(false);
	m_running.store(true);
	resetStickyState();
	m_thread = std::thread(&Aim::aimThreadFunc, this);

	blog(LOG_INFO, "[Aim] Aim thread started (StickyAim=%s)", m_stickyAimEnabled ? "ON" : "OFF");
}

void Aim::stop()
{
	if (!m_running.load()) {
		return; // 未运行
	}

	m_shouldStop.store(true);

	if (m_thread.joinable()) {
		m_thread.join();
	}

	m_running.store(false);
	resetStickyState();
	blog(LOG_INFO, "[Aim] Aim thread stopped");
}

void Aim::setTarget(float cx, float cy, float w, float h, float conf, uint64_t frame)
{
	std::lock_guard<std::mutex> lock(m_targetMutex);
	m_rawTarget.cx = cx;
	m_rawTarget.cy = cy;
	m_rawTarget.w = w;
	m_rawTarget.h = h;
	m_rawTarget.conf = conf;
	m_rawTarget.frame = frame;
	m_rawTarget.valid = true;
}

void Aim::clearTarget()
{
	std::lock_guard<std::mutex> lock(m_targetMutex);
	m_rawTarget.valid = false;
}

void Aim::setScreenSize(int width, int height)
{
	m_screenWidth = width;
	m_screenHeight = height;
}

void Aim::setHotkeys(bool leftMouse, bool rightMouse, bool xbutton1, bool xbutton2)
{
	std::lock_guard<std::mutex> lock(m_hotkeyMutex);
	m_hotkeyLeftMouse = leftMouse;
	m_hotkeyRightMouse = rightMouse;
	m_hotkeyXButton1 = xbutton1;
	m_hotkeyXButton2 = xbutton2;

	blog(LOG_INFO, "[Aim] Hotkeys updated: L=%d R=%d X1=%d X2=%d", leftMouse, rightMouse, xbutton1, xbutton2);
}

void Aim::resetStickyState()
{
	m_stickyState.hasLockedTarget = false;
	m_stickyState.velocityX = 0.0f;
	m_stickyState.velocityY = 0.0f;
	m_stickyState.framesWithoutMatch = 0;
	m_stickyState.framesWithoutDetection = 0;
	m_stickyState.lockScore = 0.0f;
	m_stickyState.lockedTarget = AimTarget{};
}

float Aim::getDistanceSq(float x1, float y1, float x2, float y2)
{
	float dx = x2 - x1;
	float dy = y2 - y1;
	return dx * dx + dy * dy;
}

bool Aim::isSameTarget(const AimTarget &a, const AimTarget &b)
{
	// 计算目标大小
	float areaA = a.w * a.h;
	float areaB = b.w * b.h;
	float sizeA = std::sqrt(areaA);

	// 跟踪半径 = 目标大小的 2 倍
	float trackingRadius = sizeA * 2.0f;
	float trackingRadiusSq = trackingRadius * trackingRadius;

	// 位置距离
	float distSq = getDistanceSq(a.cx, a.cy, b.cx, b.cy);

	// 大小相似度 (0-1, 越接近1越相似)
	float sizeRatio = (std::min)(areaA, areaB) / (std::max)(areaA, areaB + 0.001f);

	// 同一目标条件: 距离在跟踪半径内 且 大小相似度 > 0.5
	return (distSq < trackingRadiusSq) && (sizeRatio > 0.5f);
}

void Aim::updateVelocity(const AimTarget &newTarget)
{
	if (!m_stickyState.hasLockedTarget) {
		return;
	}

	// 计算新速度
	float newVelX = newTarget.cx - m_stickyState.lockedTarget.cx;
	float newVelY = newTarget.cy - m_stickyState.lockedTarget.cy;

	// EMA 平滑: velocity = old * smoothing + new * (1 - smoothing)
	m_stickyState.velocityX =
		m_stickyState.velocityX * m_velocitySmoothingFactor + newVelX * (1.0f - m_velocitySmoothingFactor);
	m_stickyState.velocityY =
		m_stickyState.velocityY * m_velocitySmoothingFactor + newVelY * (1.0f - m_velocitySmoothingFactor);
}

AimTarget Aim::getPredictedTarget(int framesAhead)
{
	AimTarget predicted = m_stickyState.lockedTarget;

	// 使用速度预测未来位置
	predicted.cx += m_stickyState.velocityX * static_cast<float>(framesAhead);
	predicted.cy += m_stickyState.velocityY * static_cast<float>(framesAhead);

	// 置信度随时间衰减
	predicted.conf *= (1.0f - static_cast<float>(framesAhead) * 0.15f);
	if (predicted.conf < 0.0f) {
		predicted.conf = 0.0f;
	}

	return predicted;
}

AimTarget Aim::handleStickyAim(const AimTarget &newDetection)
{
	// 如果粘滞瞄准未启用，直接返回新检测
	if (!m_stickyAimEnabled) {
		m_stickyState.lockedTarget = newDetection;
		m_stickyState.hasLockedTarget = newDetection.valid;
		return newDetection;
	}

	// 情况1: 无检测
	if (!newDetection.valid) {
		m_stickyState.framesWithoutDetection++;

		// 有锁定目标且在容忍期内 -> 返回预测位置
		if (m_stickyState.hasLockedTarget &&
		    m_stickyState.framesWithoutDetection <= m_maxFramesWithoutDetection) {
			m_stickyState.lockScore *= LOCK_SCORE_DECAY;
			return getPredictedTarget(m_stickyState.framesWithoutDetection);
		}

		// 超出容忍期 -> 重置
		resetStickyState();
		AimTarget invalid;
		invalid.valid = false;
		return invalid;
	}

	// 有检测，重置无检测计数
	m_stickyState.framesWithoutDetection = 0;

	// 情况2: 没有锁定目标 -> 锁定新目标
	if (!m_stickyState.hasLockedTarget) {
		m_stickyState.lockedTarget = newDetection;
		m_stickyState.hasLockedTarget = true;
		m_stickyState.velocityX = 0.0f;
		m_stickyState.velocityY = 0.0f;
		m_stickyState.lockScore = LOCK_SCORE_GAIN;
		m_stickyState.framesWithoutMatch = 0;
		return newDetection;
	}

	// 情况3: 有锁定目标，检查新检测是否是同一目标
	if (isSameTarget(m_stickyState.lockedTarget, newDetection)) {
		// 同一目标 -> 更新
		m_stickyState.framesWithoutMatch = 0;
		updateVelocity(newDetection);
		m_stickyState.lockScore = (std::min)(MAX_LOCK_SCORE, m_stickyState.lockScore + LOCK_SCORE_GAIN);
		m_stickyState.lockedTarget = newDetection;
		return newDetection;
	}

	// 情况4: 不同目标 -> 需要迟滞判断
	m_stickyState.framesWithoutMatch++;

	// 计算新目标与鼠标中心的距离
	int mouseX, mouseY;
	getMousePos(mouseX, mouseY);
	float distToNewSq =
		getDistanceSq(newDetection.cx, newDetection.cy, static_cast<float>(mouseX), static_cast<float>(mouseY));

	// 如果新目标非常接近鼠标中心 (用户明显在瞄准它) -> 快速切换
	float quickSwitchThreshold = m_stickyThreshold * 0.25f;
	bool newTargetVeryCentered = distToNewSq < (quickSwitchThreshold * quickSwitchThreshold);

	// 切换条件: 新目标非常居中 或 连续多帧未匹配
	if (newTargetVeryCentered || m_stickyState.framesWithoutMatch >= m_maxFramesWithoutMatch) {
		// 切换到新目标
		m_stickyState.lockedTarget = newDetection;
		m_stickyState.velocityX = 0.0f;
		m_stickyState.velocityY = 0.0f;
		m_stickyState.lockScore = LOCK_SCORE_GAIN;
		m_stickyState.framesWithoutMatch = 0;

		blog(LOG_INFO, "[Aim] Target switched (centered=%d, frames=%d)", newTargetVeryCentered ? 1 : 0,
		     m_stickyState.framesWithoutMatch);

		return newDetection;
	}

	// 还未准备切换 -> 返回空目标 (保持当前位置，不移动)
	// 这避免了在两个目标之间来回跳动
	AimTarget holdPosition;
	holdPosition.valid = false;
	return holdPosition;
}

void Aim::aimThreadFunc()
{
	blog(LOG_INFO, "[Aim] Thread running, update interval: %d ms", m_updateIntervalMs);

	uint64_t lastProcessedFrame = 0;

	while (!m_shouldStop.load()) {
		AimTarget rawTarget;
		AimTarget finalTarget;

		// 获取当前原始目标
		{
			std::lock_guard<std::mutex> lock(m_targetMutex);
			rawTarget = m_rawTarget;
		}

		// 只处理新的帧
		if (rawTarget.frame > lastProcessedFrame || !rawTarget.valid) {
			if (rawTarget.valid) {
				lastProcessedFrame = rawTarget.frame;
			}

			// 应用粘滞瞄准算法
			finalTarget = handleStickyAim(rawTarget);

			// 如果有有效目标且热键按下，移动鼠标
			if (finalTarget.valid && isHotkeyPressed()) {
				int mouseX, mouseY;
				getMousePos(mouseX, mouseY);

				// 计算需要移动的距离
				float dx = finalTarget.cx - static_cast<float>(mouseX);
				float dy = finalTarget.cy - static_cast<float>(mouseY);

				// 应用灵敏度和平滑
				dx *= m_sensitivity * m_smoothing;
				dy *= m_sensitivity * m_smoothing;

				// 只有当偏移大于阈值时才移动 (避免抖动)
				const float threshold = 1.0f;
				if (std::abs(dx) > threshold || std::abs(dy) > threshold) {
					int moveX = static_cast<int>(std::round(dx));
					int moveY = static_cast<int>(std::round(dy));

					moveMouse(moveX, moveY);

					// Debug 输出 (每 100 帧输出一次)
					static int debugCnt = 0;
					if (++debugCnt % 100 == 0) {
						blog(LOG_INFO,
						     "[Aim] Target: (%.1f, %.1f) Mouse: (%d, %d) Move: (%d, %d) Lock: %.1f",
						     finalTarget.cx, finalTarget.cy, mouseX, mouseY, moveX, moveY,
						     m_stickyState.lockScore);
					}
				}
			}
		}

		// 休眠
		std::this_thread::sleep_for(std::chrono::milliseconds(m_updateIntervalMs));
	}

	blog(LOG_INFO, "[Aim] Thread exiting");
}

void Aim::moveMouse(int dx, int dy)
{
	// 使用 mouse_event 进行相对移动 (Win32 API)
	mouse_event(MOUSEEVENTF_MOVE, static_cast<DWORD>(dx), static_cast<DWORD>(dy), 0, 0);
}

void Aim::getMousePos(int &x, int &y)
{
	POINT pt;
	if (GetCursorPos(&pt)) {
		x = pt.x;
		y = pt.y;
	} else {
		x = 0;
		y = 0;
	}
}

bool Aim::isHotkeyPressed()
{
	// 获取当前热键配置
	bool leftMouse, rightMouse, xbutton1, xbutton2;
	{
		std::lock_guard<std::mutex> lock(m_hotkeyMutex);
		leftMouse = m_hotkeyLeftMouse;
		rightMouse = m_hotkeyRightMouse;
		xbutton1 = m_hotkeyXButton1;
		xbutton2 = m_hotkeyXButton2;
	}

	// 检测按键状态 (GetAsyncKeyState 高位为1表示按下)
	if (leftMouse && (GetAsyncKeyState(VK_LBUTTON) & 0x8000)) {
		return true;
	}
	if (rightMouse && (GetAsyncKeyState(VK_RBUTTON) & 0x8000)) {
		return true;
	}
	if (xbutton1 && (GetAsyncKeyState(VK_XBUTTON1) & 0x8000)) {
		return true;
	}
	if (xbutton2 && (GetAsyncKeyState(VK_XBUTTON2) & 0x8000)) {
		return true;
	}

	return false;
}
