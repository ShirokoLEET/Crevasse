#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdint>
#include <chrono>

// 目标数据结构
struct AimTarget {
	float cx = 0.0f;    // 目标中心 X (屏幕坐标)
	float cy = 0.0f;    // 目标中心 Y (屏幕坐标)
	float w = 0.0f;     // 检测框宽度
	float h = 0.0f;     // 检测框高度
	float conf = 0.0f;  // 置信度
	bool valid = false; // 是否有有效目标
	uint64_t frame = 0; // 帧号
};

// 粘滞瞄准状态
struct StickyAimState {
	AimTarget lockedTarget;         // 当前锁定的目标
	float velocityX = 0.0f;         // 目标X速度 (EMA平滑)
	float velocityY = 0.0f;         // 目标Y速度 (EMA平滑)
	int framesWithoutMatch = 0;     // 连续多少帧未匹配到锁定目标
	int framesWithoutDetection = 0; // 连续多少帧无任何检测
	float lockScore = 0.0f;         // 锁定分数 (越高越稳定)
	bool hasLockedTarget = false;   // 是否有锁定目标
};

class Aim {
public:
	Aim();
	~Aim();

	// 启动/停止瞄准线程
	void start();
	void stop();

	// 设置目标 (由渲染线程调用) - 增加宽高参数用于 sticky aim
	void setTarget(float cx, float cy, float w, float h, float conf, uint64_t frame);

	// 清除目标 (无检测时调用)
	void clearTarget();

	// 设置屏幕尺寸 (用于坐标转换)
	void setScreenSize(int width, int height);

	// 瞄准是否激活
	bool isRunning() const { return m_running.load(); }

	// 配置参数
	void setStickyAimEnabled(bool enabled) { m_stickyAimEnabled = enabled; }
	void setStickyThreshold(float threshold) { m_stickyThreshold = threshold; }
	void setSensitivity(float sensitivity) { m_sensitivity = sensitivity; }
	void setSmoothing(float smoothing) { m_smoothing = smoothing; }

	// 热键配置
	void setHotkeys(bool leftMouse, bool rightMouse, bool xbutton1, bool xbutton2);

private:
	// 工作线程函数
	void aimThreadFunc();

	// 粘滞瞄准算法 - 返回最终瞄准目标
	AimTarget handleStickyAim(const AimTarget &newDetection);

	// 判断两个检测是否是同一目标
	bool isSameTarget(const AimTarget &a, const AimTarget &b);

	// 更新目标速度 (EMA)
	void updateVelocity(const AimTarget &newTarget);

	// 获取预测位置 (基于速度)
	AimTarget getPredictedTarget(int framesAhead);

	// 重置粘滞瞄准状态
	void resetStickyState();

	// 获取距离平方
	float getDistanceSq(float x1, float y1, float x2, float y2);

	// 使用 Win32 API 移动鼠标
	void moveMouse(int dx, int dy);

	// 获取当前鼠标位置
	void getMousePos(int &x, int &y);

	// 检测热键是否按下
	bool isHotkeyPressed();

	// 线程相关
	std::thread m_thread;
	std::atomic<bool> m_running{false};
	std::atomic<bool> m_shouldStop{false};

	// 目标数据 (需要互斥保护)
	std::mutex m_targetMutex;
	AimTarget m_rawTarget;        // 原始检测目标
	StickyAimState m_stickyState; // 粘滞瞄准状态

	// 屏幕尺寸
	int m_screenWidth = 1920;
	int m_screenHeight = 1080;

	// 瞄准参数
	float m_sensitivity = 1.0f; // 灵敏度
	float m_smoothing = 0.5f;   // 平滑系数 (0-1, 越小越平滑)
	int m_updateIntervalMs = 2; // 更新间隔 (毫秒)

	// 粘滞瞄准参数
	bool m_stickyAimEnabled = true;         // 是否启用粘滞瞄准
	float m_stickyThreshold = 50.0f;        // 切换阈值 (像素)
	int m_maxFramesWithoutMatch = 3;        // 切换需要连续未匹配帧数
	int m_maxFramesWithoutDetection = 5;    // 最大丢失容忍帧数
	float m_velocitySmoothingFactor = 0.7f; // 速度EMA平滑系数

	// 热键配置 (需要互斥保护)
	std::mutex m_hotkeyMutex;
	bool m_hotkeyLeftMouse = false; // 鼠标左键
	bool m_hotkeyRightMouse = true; // 鼠标右键 (默认)
	bool m_hotkeyXButton1 = false;  // 侧键1
	bool m_hotkeyXButton2 = false;  // 侧键2

	// 锁定分数相关
	static constexpr float LOCK_SCORE_GAIN = 2.0f;
	static constexpr float LOCK_SCORE_DECAY = 0.8f;
	static constexpr float MAX_LOCK_SCORE = 10.0f;
};
