# YouTube Data API v3 Setup Guide

This guide explains how to obtain and configure a YouTube Data API v3 key to enable
`likes` and `views` population in the `VideoMetadata` DTO.

---

## 1. Create a Google Cloud Project

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Click **Select a project** → **New Project**
3. Give it a name (e.g. `yt-rag-app`) and click **Create**

---

## 2. Enable the YouTube Data API v3

1. In the Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for **YouTube Data API v3**
3. Click on it and press **Enable**

---

## 3. Create an API Key

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **API Key**
3. Copy the generated key
4. (Recommended) Click **Restrict Key** and limit it to the YouTube Data API v3 only

---

## 4. Configure the Project

Add the key to your `.env` file:

```
YOUTUBE_API_KEY=your-api-key-here
```

---

## 5. Quota

The free tier provides **10,000 units per day**.  
Each `videos.list` call (used to fetch likes and views) costs **1 unit**.  
This is sufficient for production use at normal scale.

---

## 6. What Changes in Code

Once the key is in `.env`, uncomment and implement the `_fetch_video_stats`
method in `src/rag/loaders/youtube_loader.py`. It will call:

```
GET https://www.googleapis.com/youtube/v3/videos
  ?part=statistics
  &id={video_id}
  &key={YOUTUBE_API_KEY}
```

The response provides:

| Field          | Maps to              |
|----------------|----------------------|
| `viewCount`    | `VideoMetadata.views` |
| `likeCount`    | `VideoMetadata.likes` |

> **Note:** YouTube removed public dislike counts from the API in December 2021.
> Only `likeCount` is available.