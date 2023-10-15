import { Component } from '@angular/core';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent {
  selectedFile: File | null = null;

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    this.selectedFile = file;
  }

  uploadFile() {
    if (this.selectedFile) {
      // Perform file upload logic here (e.g., using an HTTP request).
      console.log('Uploading file:', this.selectedFile);
    }
  }
}
