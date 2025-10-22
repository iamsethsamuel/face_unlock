import { ThemedText } from "@/components/themed-text";
import { ThemedView } from "@/components/themed-view";
import { Camera, CameraView } from "expo-camera";
import { router, useLocalSearchParams } from "expo-router";
import * as SecureStore from "expo-secure-store";
import React, { useEffect, useRef, useState } from "react";
import { Animated, Dimensions, Easing, StyleSheet, View } from "react-native";

const USERS_KEY = "enrolled_users";
const { width } = Dimensions.get("window");
const CIRCLE_SIZE = width * 0.6;
const ANIMATION_DURATION = 10000; // 10 seconds

export default function EnrollScreen() {
    const { name } = useLocalSearchParams();
    const [hasPermission, setHasPermission] = useState<boolean | null>(null);
    const cameraRef = useRef<any>(null);
    const animation = useRef(new Animated.Value(0)).current;
    const [recording, setRecording] = useState(false);
    const [countdown, setCountdown] = useState(10);

    useEffect(() => {
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            setHasPermission(status === "granted");
        })();
    }, []);

    useEffect(() => {
        if (hasPermission && cameraRef.current && name && !recording) {
            const startAnimationLocal = () => {
                animation.setValue(0);
                Animated.loop(
                    Animated.timing(animation, {
                        toValue: 1,
                        duration: ANIMATION_DURATION,
                        easing: Easing.linear,
                        useNativeDriver: true,
                    }),
                    { iterations: 1 }
                ).start();
            };

            const doEnrollment = async () => {
                setRecording(true);
                startAnimationLocal();
                startCountdown();

                try {
                    const video = await cameraRef.current?.recordAsync({
                        maxDuration: 10,
                    });

                    if (video) {
                        console.log("Video recorded:", video.uri);
                        await saveUser(name as string);
                        router.back();
                    }
                } catch (e) {
                    console.error("Enrollment error", e);
                } finally {
                    setRecording(false);
                }
            };

            void doEnrollment();
        }
    }, [hasPermission, name, recording, animation]);

    // startAnimation was inlined into the enrollment effect; remove unused function

    const startCountdown = () => {
        let count = 10;
        setCountdown(count);
        const interval = setInterval(() => {
            count--;
            setCountdown(count);
            if (count === 0) {
                clearInterval(interval);
            }
        }, 1000);
    };

    const translateX = animation.interpolate({
        inputRange: [0, 0.25, 0.5, 0.75, 1],
        outputRange: [0, CIRCLE_SIZE / 2 - 15, 0, -(CIRCLE_SIZE / 2 - 15), 0],
    });

    const translateY = animation.interpolate({
        inputRange: [0, 0.25, 0.5, 0.75, 1],
        outputRange: [-(CIRCLE_SIZE / 2 - 15), 0, CIRCLE_SIZE / 2 - 15, 0, -(CIRCLE_SIZE / 2 - 15)],
    });

    const saveUser = async (userName: string) => {
        const usersJson = await SecureStore.getItemAsync(USERS_KEY);
        let users: string[] = usersJson ? JSON.parse(usersJson) : [];
        if (!users.includes(userName)) {
            users.push(userName);
            await SecureStore.setItemAsync(USERS_KEY, JSON.stringify(users));
        }
    };

    // Resolve the runtime component shape in a typed way: some bundlers expose the component on `.default`.
    // CameraComponent is typed as a React.ComponentType to make it usable in JSX while keeping the ref typed.
    const _cameraAsComponent = Camera as unknown as React.ComponentType<any>;
    const CameraComponent: React.ComponentType<any> = (_cameraAsComponent as any).default ?? _cameraAsComponent;

    if (hasPermission === null) {
        return (
            <ThemedView style={styles.container}>
                <ThemedText>Requesting camera permission...</ThemedText>
            </ThemedView>
        );
    }
    if (hasPermission === false) {
        return (
            <ThemedView style={styles.container}>
                <ThemedText>No access to camera</ThemedText>
            </ThemedView>
        );
    }

    return (
        <ThemedView style={styles.container}>
            {/* <CameraComponent style={styles.camera} type={'front'} ref={cameraRef}> */}
            <CameraView style={styles.camera}  facing="front" />

            <View style={styles.overlay}>
                <View style={styles.circleContainer}>
                    <View style={styles.circleCutout} />
                    <Animated.View
                        style={[
                            styles.animationDot,
                            {
                                transform: [{ translateX }, { translateY }],
                            },
                        ]}
                    />
                </View>
                <ThemedText style={styles.instructionText}>
                    {/* {`Move your head in a circle for ${countdown} seconds`} */}
                </ThemedText>
                <ThemedText style={styles.userNameText}>Enrolling: {name}</ThemedText>
            </View>
            {/* </CameraComponent> */}
        </ThemedView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
    },
    camera: {
        flex: 1,
        width: "100%",
        height: "100%",
    },
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.5)", // Semi-transparent overlay
        justifyContent: "center",
        alignItems: "center",
    },
    circleContainer: {
        width: CIRCLE_SIZE,
        height: CIRCLE_SIZE,
        borderRadius: CIRCLE_SIZE / 2,
        borderWidth: 3,
        borderColor: "white",
        justifyContent: "center",
        alignItems: "center",
        overflow: "hidden",
    },
    circleCutout: {
        width: CIRCLE_SIZE - 6, // Slightly smaller to show border
        height: CIRCLE_SIZE - 6,
        borderRadius: (CIRCLE_SIZE - 6) / 2,
        backgroundColor: "transparent",
        position: "absolute",
    },
    animationDot: {
        width: 30,
        height: 30,
        borderRadius: 15,
        backgroundColor: "#007AFF", // Blue dot
        position: "absolute",
    },
    instructionText: {
        fontSize: 20,
        fontWeight: "bold",
        color: "white",
        marginTop: 20,
        textAlign: "center",
    },
    userNameText: {
        fontSize: 16,
        color: "white",
        marginTop: 10,
    },
});
