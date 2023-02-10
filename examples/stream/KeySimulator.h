#pragma once

#include <set>

class KeySimulator
{
public:
    KeySimulator();
    virtual ~KeySimulator();

    // Presses a key specified by the given key code
    virtual void PressKey(char c) = 0;
    // Releases a key specified by the given key code
    virtual void ReleaseKey(char c) = 0;
    // Simulates typing the given text
    void TypePhrase(const char* text);

protected:
    std::set<int> pressedKeys;
};

class WindowsKeySimulator : public KeySimulator
{
public:
    WindowsKeySimulator();
    virtual ~WindowsKeySimulator();

    // Presses a key specified by the given key code using the Windows API
    virtual void PressKey(char c) override;
    // Releases a key specified by the given key code using the Windows API
    virtual void ReleaseKey(char c) override;
};