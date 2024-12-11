async function handleImageUpload(input) {
  if (input.files && input.files[0]) {
    const formData = new FormData();
    formData.append('image', input.files[0]);

    try {
      const response = await fetch('/api/image-search', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const results = await response.json();
        // Redirect to search results page with the image query ID
        window.location.href = `/search?image_query=${results.query_id}`;
      }
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  }
}
