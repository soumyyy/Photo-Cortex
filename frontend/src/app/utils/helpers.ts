interface ImageAnalysis {
  filename: string;
  faces: Array<{
    confidence: number;
    bbox: number[];
    person_id?: string;
  }>;
}

interface Person {
  faces: Array<{
    confidence: number;
    bbox: number[];
    filename: string;
  }>;
  name: string;
  totalPhotos: number;
}

export function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

export function groupFacesByPerson(images: ImageAnalysis[]): Person[] {
  const peopleMap = new Map<string, Person>();
  
  images.forEach(image => {
    if (image.faces) {
      image.faces.forEach((face) => {
        const personId = face.person_id || 'unknown';
        if (!peopleMap.has(personId)) {
          peopleMap.set(personId, {
            faces: [],
            name: `Person ${peopleMap.size + 1}`,
            totalPhotos: 0
          });
        }
        
        const person = peopleMap.get(personId)!;
        person.faces.push({
          ...face,
          filename: image.filename
        });
        person.totalPhotos++;
      });
    }
  });
  
  return Array.from(peopleMap.values());
}