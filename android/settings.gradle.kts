pluginManagement {
    repositories {
        google()
        // Maven Central sometimes redirects large artifacts to GitHub, which intermittently
        // answers 504. Google's official Maven Central mirror serves the same artifacts directly.
        maven("https://maven-central.storage-download.googleapis.com/maven2")
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        maven("https://maven-central.storage-download.googleapis.com/maven2")
        mavenCentral()
    }
}
rootProject.name = "AIPlaygroundAndroid"
include(":app")
