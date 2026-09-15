import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}
val releaseVersion = Properties().apply {
    rootProject.file("version.properties").inputStream().use { load(it) }
}
android {
    namespace = "com.minashin1120.aiplayground"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.minashin1120.aiplayground"
        minSdk = 26
        targetSdk = 36
        versionCode = releaseVersion.getProperty("VERSION_CODE").toInt()
        versionName = releaseVersion.getProperty("VERSION_NAME")
        buildConfigField("String", "BASE_URL", "\"https://ai.minashin1120.com/\"")
    }
    signingConfigs {
        create("sharedDebug") {
            val fixedKey = rootProject.file("ci/debug.keystore")
            if (!fixedKey.isFile || fixedKey.length() == 0L) {
                throw GradleException("ci/debug.keystore is missing or empty. Run Android CI on main once; never generate a replacement locally.")
            }
            storeFile = fixedKey
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }
    buildTypes {
        debug { signingConfig = signingConfigs.getByName("sharedDebug") }
        release {
            signingConfig = signingConfigs.getByName("sharedDebug")
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true; buildConfig = true }
    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    lint { abortOnError = true; checkReleaseBuilds = true }
}
dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.08.00"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.2")
    implementation("androidx.browser:browser:1.9.0")
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("com.squareup.okhttp3:okhttp:5.3.0")
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    testImplementation("com.squareup.okhttp3:mockwebserver3:5.3.0")
}
