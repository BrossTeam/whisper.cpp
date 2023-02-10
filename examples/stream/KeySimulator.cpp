#include "KeySimulator.h"
#include <Windows.h>

KeySimulator::KeySimulator() {}
KeySimulator::~KeySimulator() {}

void KeySimulator::TypePhrase(const char* text)
{
    for (int i = 0; text[i]; i++)
    {
        PressKey(text[i]);
        ReleaseKey(text[i]);
    }
}

WindowsKeySimulator::WindowsKeySimulator() {}
WindowsKeySimulator::~WindowsKeySimulator() {}

void WindowsKeySimulator::PressKey(char c)
{
    if (pressedKeys.count(c) > 0)
    {
        return;
    }

    // Create the input structure for the 'KEYEVENTF_UNICODE' flag
    INPUT input = { 0 };
    input.type = INPUT_KEYBOARD;
    input.ki.dwFlags = KEYEVENTF_UNICODE;

    bool isUppercase = isupper(c);

    if (isUppercase)
    {
        // Simulate pressing the 'SHIFT' key
        input.ki.wVk = VK_SHIFT;
        SendInput(1, &input, sizeof(INPUT));
    }

    // Simulate pressing the key
    input.ki.wVk = 0;
    input.ki.wScan = c;
    SendInput(1, &input, sizeof(INPUT));

    pressedKeys.insert(c);
}

void WindowsKeySimulator::ReleaseKey(char c)
{
    if (pressedKeys.count(c) == 0)
    {
        return;
    }

    // Create the input structure for the 'KEYEVENTF_UNICODE' flag
    INPUT input = { 0 };
    input.type = INPUT_KEYBOARD;
    input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;

    // Simulate releasing the key
    input.ki.wVk = 0;
    input.ki.wScan = c;
    SendInput(1, &input, sizeof(INPUT));

    bool isUppercase = isupper(c);

    if (isUppercase)
    {
        // Simulate releasing the 'SHIFT' key
        input.ki.wVk = VK_SHIFT;
        input.ki.wScan = 0;
        SendInput(1, &input, sizeof(INPUT));
    }

    pressedKeys.erase(c);
}
